"""
engine/memory_system.py — Sistem memori berlapis yang dioptimasi.

Perubahan dari versi sebelumnya:
1. EpisodicMemory: tambah key_facts per sesi (rule-based, tanpa LLM)
2. CoreMemory: update async (background thread, tidak blocking)
3. HybridMemory: gabungan core + episodic retrieval yang efisien
4. Embedding model tetap sama tapi loading lebih efisien
"""

import json
import re
import threading
from pathlib import Path
import numpy as np
import datetime
import torch
from transformers import AutoTokenizer, AutoModel

# ─── Embedding Model ───────────────────────────────────────────────────────────

HF_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_MODEL_PATH = Path("model") / "embedding_model" / HF_MODEL_NAME.split("/")[-1]

def _load_embedding_model():
    if LOCAL_MODEL_PATH.exists():
        tok = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        mdl = AutoModel.from_pretrained(LOCAL_MODEL_PATH)
    else:
        tok = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        mdl = AutoModel.from_pretrained(HF_MODEL_NAME)
        LOCAL_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(LOCAL_MODEL_PATH)
        mdl.save_pretrained(LOCAL_MODEL_PATH)
    return tok, mdl

_tokenizer, _model = _load_embedding_model()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(
        mask_expanded.sum(1), min=1e-9
    )


def create_embedding(text: str) -> np.ndarray:
    if not text or not text.strip():
        return np.zeros(_model.config.hidden_size)
    encoded = _tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = _model(**encoded)
    emb = mean_pooling(out, encoded["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb[0].cpu().numpy()


# ─── Key Facts Extractor (Rule-Based, Tanpa LLM) ──────────────────────────────

# Pola untuk mendeteksi fakta penting dari percakapan
_FACT_PATTERNS = [
    # Rencana / tujuan
    (r"\b(mau|pengen|rencana|besok|minggu depan|nanti)\b.{5,60}", "rencana"),
    # Lokasi / tempat
    (r"\b(ke|di|dari)\s+(jepang|bali|jakarta|pantai|gunung|bandara|rumah)\b.{0,40}", "lokasi"),
    # Preferensi makanan/minuman
    (r"\b(suka|favorit|enak|makan|minum)\s+\w+", "preferensi"),
    # Emosi / perasaan kuat
    (r"\b(sayang|cinta|kangen|rindu|bahagia|seneng)\b.{0,30}", "emosi"),
    # Fakta tentang user
    (r"\baku\s+(lagi|udah|baru|mau)\s+\w+.{0,40}", "aktivitas_user"),
]

def extract_key_facts(conversation: list) -> list:
    """
    Ekstrak fakta kunci dari percakapan menggunakan regex pattern matching.
    Jauh lebih cepat dari LLM summarization.
    """
    facts = []
    seen = set()

    for msg in conversation:
        if msg["role"] not in ("user", "assistant"):
            continue
        text = msg["content"].lower().strip()
        if not text or len(text) < 10:
            continue

        for pattern, category in _FACT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                fact = match.group(0).strip()
                # Deduplicate dan filter terlalu pendek
                key = fact[:40]
                if key not in seen and len(fact) > 8:
                    seen.add(key)
                    facts.append({"category": category, "fact": fact})
                if len(facts) >= 15:  # batas per sesi
                    break

    return facts


def facts_to_text(facts: list) -> str:
    """Konversi list facts ke teks ringkas untuk diinjeksi ke context."""
    if not facts:
        return ""
    by_cat = {}
    for f in facts:
        by_cat.setdefault(f["category"], []).append(f["fact"])
    lines = []
    for cat, items in by_cat.items():
        lines.append(f"[{cat}] " + "; ".join(items[:3]))
    return "\n".join(lines)


# ─── Base Memory ───────────────────────────────────────────────────────────────

class BaseMemory:
    def __init__(self, file_path: Path, default_content):
        self.file_path = file_path
        self._default = default_content
        self._lock = threading.Lock()
        self.data = self._load()

    def _load(self):
        if not self.file_path.exists() or self.file_path.stat().st_size == 0:
            self._write(self._default)
            return self._default
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            print(f"[Memory] File {self.file_path.name} rusak, reset.")
            self._write(self._default)
            return self._default

    def _write(self, data):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save(self):
        with self._lock:
            self._write(self.data)

    def save_async(self):
        """Simpan di background thread agar tidak blocking."""
        t = threading.Thread(target=self.save, daemon=True)
        t.start()
        return t


# ─── Semantic Memory ───────────────────────────────────────────────────────────

class SemanticMemory(BaseMemory):
    def __init__(self, directory: Path):
        super().__init__(directory / "semantic.json", default_content={})

    def add_fact(self, key: str, value):
        if self.data.get(key) != value:
            self.data[key] = value
            self.save_async()

    def get_fact(self, key: str):
        return self.data.get(key)

    def get_all_facts(self) -> dict:
        return self.data.copy()


# ─── Episodic Memory (Hybrid: embedding + key facts) ──────────────────────────

class EpisodicMemory(BaseMemory):
    def __init__(self, directory: Path):
        super().__init__(directory / "episodic.json", default_content=[])

    def add(self, conversation: list, llm_summary: str = ""):
        """
        Simpan sesi percakapan dengan:
        - embedding untuk semantic search
        - key_facts untuk retrieval cepat tanpa LLM
        - llm_summary opsional (diisi background jika ada)
        """
        text_conv = " ".join(
            f"{m['role']}: {m['content']}"
            for m in conversation
            if m["role"] in ("user", "assistant") and m["content"]
        )
        embedding = create_embedding(text_conv).tolist()
        key_facts = extract_key_facts(conversation)

        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "key_facts": key_facts,
            "llm_summary": llm_summary,   # "" jika belum diproses
            "embedding": embedding,
            # Simpan conversation ringkas (hanya user+assistant, bukan system)
            "conversation": [
                m for m in conversation
                if m["role"] in ("user", "assistant") and m["content"]
            ],
        }
        self.data.append(entry)
        # Batasi total sesi yang disimpan (jaga file tidak terlalu besar)
        if len(self.data) > 50:
            self.data = self.data[-50:]
        self.save_async()
        print(f"[Episodic] Sesi disimpan. {len(key_facts)} key facts diekstrak.")

    def update_summary(self, session_index: int, summary: str):
        """Update LLM summary untuk sesi tertentu (dipanggil dari background thread)."""
        if 0 <= session_index < len(self.data):
            self.data[session_index]["llm_summary"] = summary
            self.save_async()

    def search(self, query: str, top_k: int = 3, threshold: float = 0.15) -> list:
        """Semantic search berdasarkan embedding similarity."""
        if not self.data:
            return []
        q_emb = create_embedding(query)
        sims = [np.dot(q_emb, np.array(m["embedding"])) for m in self.data]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [self.data[i] for i in top_idx if sims[i] > threshold]

    def get_last_n(self, n: int = 3) -> list:
        return self.data[-n:]

    def get_recent_facts_text(self, n_sessions: int = 3, max_facts: int = 12) -> str:
        """
        Ambil key facts dari N sesi terakhir sebagai teks ringkas.
        Jauh lebih cepat dari LLM summarization.
        """
        sessions = self.get_last_n(n_sessions)
        all_facts = []
        for s in sessions:
            all_facts.extend(s.get("key_facts", []))
        # Prioritaskan: rencana > lokasi > preferensi > lainnya
        priority = {"rencana": 0, "lokasi": 1, "preferensi": 2, "emosi": 3, "aktivitas_user": 4}
        all_facts.sort(key=lambda f: priority.get(f.get("category", ""), 5))
        selected = all_facts[:max_facts]
        return facts_to_text(selected)


# ─── Core Memory ───────────────────────────────────────────────────────────────

class CoreMemory(BaseMemory):
    def __init__(self, directory: Path):
        super().__init__(directory / "core_memory.json", default_content={"summary": ""})

    def get_summary(self) -> str:
        return self.data.get("summary", "")

    def update_summary(self, text: str, async_save: bool = True):
        if self.data.get("summary") != text:
            self.data["summary"] = text
            if async_save:
                self.save_async()
            else:
                self.save()
            print("[Core Memory] Ringkasan diperbarui.")


# ─── Hybrid Memory (Gabungan) ──────────────────────────────────────────────────

class HybridMemory:
    """
    Lapisan abstraksi yang menggabungkan Core + Episodic memory
    dan menghasilkan context string yang siap diinjeksi ke model.

    Strategi:
    1. Core memory summary (kompak, ~100-200 kata)
    2. Key facts dari 3 sesi terakhir (rule-based, cepat)
    3. Jika ada query relevan, tambahkan episodic retrieval
    """

    def __init__(self, episodic: EpisodicMemory, core: CoreMemory):
        self.episodic = episodic
        self.core = core

    def get_context(self, current_query: str = "", max_chars: int = 1200) -> str:
        """
        Bangun string konteks memori untuk diinjeksi.
        max_chars: batas karakter kasar agar tidak melebihi token budget.
        """
        parts = []

        # 1. Core memory
        core_text = self.core.get_summary()
        if core_text:
            parts.append(f"[Inti Memori]\n{core_text}")

        # 2. Key facts terbaru (cepat, tanpa LLM)
        facts_text = self.episodic.get_recent_facts_text(n_sessions=3, max_facts=10)
        if facts_text:
            parts.append(f"[Fakta Terbaru]\n{facts_text}")

        # 3. Semantic retrieval jika ada query
        if current_query:
            recalled = self.episodic.search(current_query, top_k=1, threshold=0.2)
            for r in recalled:
                if r.get("llm_summary"):
                    parts.append(f"[Memori Relevan]\n{r['llm_summary']}")
                    break

        full_text = "\n\n".join(parts)
        # Potong jika terlalu panjang
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "..."

        return full_text

    def update_core_async(self, llm_callable, current_session_text: str):
        """
        Update core memory menggunakan LLM di background thread.
        Tidak blocking interface sama sekali.
        """
        def _worker():
            old_summary = self.core.get_summary()
            combined = ""
            if old_summary:
                combined += f"Ringkasan sebelumnya:\n{old_summary}\n\n"
            combined += f"Percakapan terbaru:\n{current_session_text}"

            prompt = (
                "Berdasarkan ringkasan sebelumnya dan percakapan terbaru, "
                "buat satu paragraf ringkas (maks 150 kata) tentang fakta penting pengguna. "
                "Fokus pada: nama, preferensi, rencana, kegiatan terkini. "
                "Tulis dalam bahasa Indonesia.\n\n"
                f"{combined}\n\nRingkasan:"
            )
            try:
                result = llm_callable(
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.1,
                    stop=["\n\n", "###"],
                )
                summary = result["choices"][0]["text"].strip()
                if summary:
                    self.core.update_summary(summary, async_save=True)
                    print("[Core Memory] Background update selesai.")
            except Exception as e:
                print(f"[Core Memory] Background update gagal: {e}")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return t
