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


def _is_zero_embedding(embedding: list) -> bool:
    """FIX #2: Cek apakah embedding semua nol — sesi kosong/invalid."""
    if not embedding:
        return True
    return np.allclose(np.array(embedding[:10]), 0.0)  # cek 10 elemen pertama saja


# ─── Key Facts Extractor (Fix #5: Diperketat) ─────────────────────────────────

_FACT_PATTERNS = [
    (r"\baku\s+suka\s+([a-zA-Z ]{4,40})", "preferensi"),
    (r"\baku\s+gak\s+suka\s+([a-zA-Z ]{4,40})", "preferensi_tidak"),
    (r"\b(mau|pengen)\s+(ke|pergi|nikah|menikah|liburan)\s+\w+.{0,30}", "rencana"),
    (r"\b(besok|minggu depan|nanti)\s+(kita|aku)\s+\w+.{0,40}", "rencana"),
    (r"\bkita\s+(ke|di)\s+(jepang|bali|jakarta|bandung|surabaya|pantai|gunung)\b.{0,25}", "lokasi"),
    (r"\baku\s+(tinggal|kerja|kuliah)\s+di\s+\w+", "identitas"),
]

_NOISE_RE = re.compile(r"^\s*\*\w+\*\s*$|[*]{2,}")

def extract_key_facts(conversation: list) -> list:
    facts = []
    seen = set()

    for msg in conversation:
        if msg["role"] != "user":
            continue
        text = msg["content"].strip()
        if not text or len(text) < 15 or _NOISE_RE.search(text):
            continue

        text_lower = text.lower()
        for pattern, category in _FACT_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                fact = match.group(0).strip()
                key = fact[:50]
                if key not in seen and len(fact) >= 12:
                    seen.add(key)
                    facts.append({
                        "category": category,
                        "fact": fact,
                        "source": text[:100],
                    })
                if len(facts) >= 10:
                    return facts
    return facts


def facts_to_text(facts: list) -> str:
    if not facts:
        return ""
    by_cat = {}
    for f in facts:
        by_cat.setdefault(f["category"], []).append(f["fact"])
    lines = []
    for cat, items in by_cat.items():
        lines.append(f"[{cat}] " + "; ".join(items[:2]))
    return "\n".join(lines)


def _build_fallback_summary(conversation: list, key_facts: list, max_chars: int = 240) -> str:
    """Ringkasan minimal saat llm_summary tidak tersedia."""
    if key_facts:
        raw = "; ".join(f.get("fact", "").strip() for f in key_facts if f.get("fact"))
        raw = re.sub(r"\s+", " ", raw).strip(" ;")
        if raw:
            return raw[:max_chars]

    user_msgs = [
        m.get("content", "").strip()
        for m in conversation
        if m.get("role") == "user" and m.get("content")
    ]
    if not user_msgs:
        return ""

    candidate = user_msgs[-1] if len(user_msgs[-1]) >= 12 else user_msgs[0]
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate[:max_chars]


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


# ─── Episodic Memory ──────────────────────────────────────────────────────────

class EpisodicMemory(BaseMemory):
    def __init__(self, directory: Path):
        super().__init__(directory / "episodic.json", default_content=[])

    def add(self, conversation: list, llm_summary: str = ""):
        text_conv = " ".join(
            f"{m['role']}: {m['content']}"
            for m in conversation
            if m["role"] in ("user", "assistant") and m["content"]
        )
        if not text_conv.strip():
            print("[Episodic] Sesi kosong, tidak disimpan.")
            return

        embedding = create_embedding(text_conv).tolist()
        key_facts = extract_key_facts(conversation)
        final_summary = (llm_summary or "").strip() or _build_fallback_summary(conversation, key_facts)

        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "key_facts": key_facts,
            "llm_summary": final_summary,
            "embedding": embedding,
            "conversation": [
                m for m in conversation
                if m["role"] in ("user", "assistant") and m["content"]
            ],
        }
        self.data.append(entry)
        if len(self.data) > 50:
            self.data = self.data[-50:]
        self.save_async()
        print(f"[Episodic] Sesi disimpan. {len(key_facts)} key facts diekstrak.")

    def search(self, query: str, top_k: int = 3, threshold: float = 0.10) -> list:
        if not self.data:
            return []

        q_emb = create_embedding(query)
        sims = []
        valid = []

        for mem in self.data:
            emb = mem.get("embedding", [])
            if _is_zero_embedding(emb):  # FIX #2
                continue
            sim = float(np.dot(q_emb, np.array(emb)))
            sims.append(sim)
            valid.append(mem)

        if not sims:
            return []

        top_idx = np.argsort(sims)[::-1][:top_k]
        results = [valid[i] for i in top_idx if sims[i] > threshold]

        if results:
            print(f"[Episodic] Search '{query[:40]}': {len(results)} hasil")
        return results

    def search_by_facts(self, topic: str, top_k: int = 2) -> list:
        if not self.data or not topic:
            return []

        topic_lower = topic.lower()
        keywords = [w for w in re.split(r'\W+', topic_lower) if len(w) > 2]

        scored = []
        for entry in self.data:
            if _is_zero_embedding(entry.get("embedding", [])):
                continue

            score = 0
            for kf in entry.get("key_facts", []):
                fact_text = kf.get("fact", "").lower()
                source_text = kf.get("source", "").lower()
                for kw in keywords:
                    if kw in fact_text:
                        score += 3
                    elif kw in source_text:
                        score += 1

            for msg in entry.get("conversation", []):
                if msg.get("role") == "user":
                    content = msg.get("content", "").lower()
                    for kw in keywords:
                        if kw in content:
                            score += 2

            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [e for _, e in scored[:top_k]]

        if results:
            print(f"[Episodic] Fact search '{topic[:40]}': {len(results)} hasil (skor: {[s for s,_ in scored[:top_k]]})")
        return results

    def get_last_n(self, n: int = 3) -> list:
        valid = [s for s in self.data if not _is_zero_embedding(s.get("embedding", []))]
        return valid[-n:]

    def get_recent_facts_text(self, n_sessions: int = 3, max_facts: int = 8) -> str:
        sessions = self.get_last_n(n_sessions)
        all_facts = []
        for s in sessions:
            for f in s.get("key_facts", []):
                # FIX #5: Filter fakta terlalu pendek atau kategori emosi
                if len(f.get("fact", "")) >= 12 and f.get("category") not in ("emosi",):
                    all_facts.append(f)

        priority = {"preferensi": 0, "rencana": 1, "lokasi": 2, "identitas": 3}
        all_facts.sort(key=lambda f: priority.get(f.get("category", ""), 5))
        return facts_to_text(all_facts[:max_facts])


# ─── Core Memory (Fix #3: Profil Pengguna) ────────────────────────────────────

class CoreMemory(BaseMemory):
    def __init__(self, directory: Path):
        super().__init__(
            directory / "core_memory.json",
            default_content={"summary": "", "user_profile": {}}
        )
        if "user_profile" not in self.data:
            self.data["user_profile"] = {}
            self.save_async()

    def get_summary(self) -> str:
        return self.data.get("summary", "")

    def get_profile(self) -> dict:
        return self.data.get("user_profile", {})

    def update_summary(self, text: str, async_save: bool = True):
        if self.data.get("summary") != text:
            self.data["summary"] = text
            if async_save:
                self.save_async()
            else:
                self.save()
            print("[Core Memory] Summary diperbarui.")

    def add_preference(self, preference: str):
        profile = self.data.setdefault("user_profile", {})
        prefs = profile.setdefault("preferensi", [])
        if preference not in prefs:
            prefs.append(preference)
            profile["preferensi"] = prefs[-20:]
            self.save_async()
            print(f"[Core Memory] Preferensi: {preference}")

    def get_context_text(self) -> str:
        parts = []

        summary = self.get_summary()
        if summary:
            clean = re.sub(r'\(Keterangan[^)]*\)', '', summary)
            clean = re.sub(r'\s+', ' ', clean).strip()
            if clean:
                parts.append(clean[:300])

        profile = self.get_profile()
        if profile:
            lines = []
            if profile.get("preferensi"):
                lines.append("Suka: " + ", ".join(profile["preferensi"][:5]))
            if profile.get("rencana"):
                r = profile["rencana"]
                lines.append("Rencana: " + (", ".join(r[:3]) if isinstance(r, list) else str(r)))
            if lines:
                parts.append("[Profil Pengguna]\n" + "\n".join(lines))

        return "\n\n".join(parts)


# ─── Hybrid Memory ────────────────────────────────────────────────────────────

class HybridMemory:
    def __init__(self, episodic: EpisodicMemory, core: CoreMemory):
        self.episodic = episodic
        self.core = core

    def get_context(
        self,
        current_query: str = "",
        recall_topic: str = "",
        max_chars: int = 1200,
    ) -> str:
        parts = []

        memory_intent = bool(re.search(
            r"\b(ingat|ingetin|ingatan|inget|flag\s*point\w*|kemarin|dulu|tadi|apa\s+tadi|apa\s+yang\s+aku\s+bilang|"
            r"kamu\s+ingat|siapa\s+namaku|nama\s+aku)\b",
            (current_query or ""),
            re.IGNORECASE,
        ))

        core_text = self.core.get_context_text()
        if core_text:
            parts.append(f"[Memori Inti]\n{core_text}")

        facts_text = self.episodic.get_recent_facts_text(n_sessions=3, max_facts=8)
        if facts_text:
            parts.append(f"[Fakta Terbaru]\n{facts_text}")

        recall_used = False
        if recall_topic and recall_topic.strip().lower() not in ("", "kosong", "-"):
            recalled = self.episodic.search_by_facts(recall_topic, top_k=2)
            for r in recalled:
                relevant_lines = []
                conv = r.get("conversation", [])
                keywords = [w for w in recall_topic.lower().split() if len(w) > 2]

                for i, msg in enumerate(conv):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if any(kw in content.lower() for kw in keywords):
                            relevant_lines.append(f"Aditiya: {content[:120]}")
                            # Ambil respons Asta berikutnya
                            if i + 1 < len(conv) and conv[i + 1].get("role") == "assistant":
                                relevant_lines.append(f"Asta: {conv[i + 1]['content'][:120]}")
                            if len(relevant_lines) >= 4:
                                break

                if relevant_lines:
                    parts.append(
                        f"[Ingatan: '{recall_topic}']\n" + "\n".join(relevant_lines)
                    )
                    recall_used = True
                    break

        if not recall_used and current_query and memory_intent:
            recalled = self.episodic.search(current_query, top_k=1, threshold=0.10)
            for r in recalled:
                if r.get("llm_summary"):
                    parts.append(f"[Memori Relevan]\n{r['llm_summary']}")
                    break
                conv = r.get("conversation", [])
                for i, msg in enumerate(conv):
                    if msg.get("role") == "user" and msg.get("content"):
                        snippet = [f"Aditiya: {msg['content'][:140]}"]
                        if i + 1 < len(conv) and conv[i + 1].get("role") == "assistant":
                            snippet.append(f"Asta: {conv[i + 1].get('content', '')[:140]}")
                        parts.append("[Memori Relevan]\n" + "\n".join(snippet))
                        break
                if parts and parts[-1].startswith("[Memori Relevan]"):
                    break

        full_text = "\n\n".join(parts)
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "..."
        return full_text

    def extract_and_save_preferences(self, conversation: list):
        pref_re = re.compile(r"\baku\s+suka\s+([a-zA-Z ]{4,30})", re.IGNORECASE)
        for msg in conversation:
            if msg.get("role") != "user":
                continue
            for match in pref_re.finditer(msg.get("content", "")):
                pref = match.group(1).strip().lower()
                if len(pref) >= 4 and pref not in ("kamu", "asta", "sama", "banget", "juga"):
                    self.core.add_preference(pref)

    def update_core_async(self, llm_callable, current_session_text: str):
        def _worker():
            old_summary = self.core.get_summary()
            combined = ""
            if old_summary:
                clean = re.sub(r'\(Keterangan[^)]*\)', '', old_summary).strip()
                combined += f"Ringkasan sebelumnya:\n{clean[:400]}\n\n"
            combined += f"Percakapan terbaru:\n{current_session_text[:800]}"

            prompt = (
                "Berdasarkan ringkasan sebelumnya dan percakapan terbaru, "
                "buat satu paragraf ringkas (maks 100 kata) tentang fakta penting pengguna. "
                "Fokus: nama, preferensi, rencana konkret, hubungan dengan Asta. "
                "Bahasa Indonesia. JANGAN tambahkan keterangan atau catatan.\n\n"
                f"{combined}\n\nRingkasan:"
            )
            try:
                result = llm_callable(
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.1,
                    stop=["\n\n", "###", "(Keterangan"],
                )
                summary = result["choices"][0]["text"].strip()
                if summary:
                    self.core.update_summary(summary, async_save=True)
                    print("[Core Memory] Background update selesai.")
            except Exception as e:
                print(f"[Core Memory] Background update gagal: {e}")

        thread = threading.Thread(target=_worker, daemon=False)
        thread.start()
        return thread
