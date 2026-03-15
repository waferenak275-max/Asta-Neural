"""
engine/thought.py — Internal Thought Engine untuk Asta.
"""

from typing import Optional

_THOUGHT_SYSTEM_TEMPLATE = """Kamu adalah sistem analisis input untuk percakapan antara DUA pihak:
- Asta      : AI perempuan, asisten sekaligus pasangan romantis {user_name}
- {user_name}: pengguna yang berbicara dengan Asta

PENTING: Dalam input di bawah, "aku" dari user = {user_name}, BUKAN Asta.
Pertanyaan seperti "aku suka apa", "aku pernah bilang apa" = tentang {user_name}.

ATURAN NEED_SEARCH — jawab yes jika SALAH SATU kondisi ini terpenuhi:
1. {user_name} meminta info faktual dari internet: cuaca, harga, kurs, berita, siapa menjabat
2. {user_name} menyebut kata "cari", "cek", "verifikasi", "search"
3. {user_name} menyebut judul lagu, film, buku, anime, game, atau karya spesifik yang
   mungkin tidak dikenal luas — model harus search untuk memastikan, BUKAN mengarang
4. {user_name} menanyakan detail tentang karya/topik yang model tidak yakin 100% kebenarannya

Jika ragu apakah model benar-benar tahu → pilih yes dan search.

ATURAN RECALL_TOPIC:
- Isi HANYA jika topiknya pernah dibahas di percakapan sebelumnya dengan {user_name}
- WAJIB gunakan nama "{user_name}" jika topiknya tentang user
- Benar : "minuman favorit {user_name}", "rencana {user_name} ke Bali"
- Salah : "minuman favorit Asta" (kecuali tentang Asta)
- DILARANG: mengisi RECALL_TOPIC dengan fakta yang dikarang/diasumsikan sendiri
- Kosongkan jika tidak ada ingatan nyata yang perlu dipanggil

FORMAT WAJIB (berhenti tepat setelah NOTE):
NEED_SEARCH: yes/no
SEARCH_QUERY: <query internet jika yes, kosong jika no>
RECALL_TOPIC: <topik ingatan nyata dari percakapan, kosong jika tidak ada>
USER_EMOTION: netral/sedih/cemas/marah/senang/romantis
EMOTION_CONFIDENCE: rendah/sedang/tinggi
TONE: romantic/casual/informative
NOTE: <catatan singkat maks 10 kata>\
"""

# Diisi dengan user_name saat runtime
_THOUGHT_SYSTEM = _THOUGHT_SYSTEM_TEMPLATE

_STOP_TOKENS = ["\n\n", "---", "Input pengguna:", "Analisis:", "</thought>", "###"]


def run_thought_pass(
    llm,
    user_input: str,
    memory_context: str,
    recent_context: str = "",
    web_search_enabled: bool = True,
    max_tokens: int = 100,
    user_name: str = "Aditiya",  # ← nama user untuk konteks identitas
    emotion_state: str = "",
) -> dict:
    # Isi template dengan nama user agar thought tahu siapa lawan bicaranya
    thought_system = _THOUGHT_SYSTEM_TEMPLATE.format(user_name=user_name)

    mem_hint = ""
    if memory_context:
        first_line = memory_context.strip().splitlines()[0]
        mem_hint = first_line[:100]

    context_block = ""
    if recent_context:
        context_block = f"Konteks percakapan terkini:\n{recent_context}\n\n"

    emotion_block = ""
    if emotion_state:
        emotion_block = f"State emosi pengguna saat ini:\n{emotion_state}\n\n"

    prompt = (
        f"{thought_system}\n\n"
        f"Hint memori: {mem_hint or '(kosong)'}\n"
        f"Web search: {'tersedia' if web_search_enabled else 'tidak tersedia'}\n\n"
        f"{context_block}"
        f"{emotion_block}"
        f"Input {user_name}: \"{user_input}\"\n\n"
        f"NEED_SEARCH:"
    )

    try:
        result = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.05,
            top_p=0.9,
            stop=_STOP_TOKENS,
        )
        raw = "NEED_SEARCH:" + result["choices"][0]["text"].strip()
    except Exception as e:
        print(f"[Thought] Pass gagal: {e}")
        raw = ""

    parsed = _parse_thought(raw)
    parsed["raw"] = raw
    return parsed


def _parse_thought(raw: str) -> dict:
    result = {
        "need_search": False,
        "search_query": "",
        "recall_topic": "",
        "user_emotion": "netral",
        "emotion_confidence": "rendah",
        "tone": "romantic",
        "note": "",
        "raw": raw,
    }
    if not raw:
        return result

    found_note = False
    for line in raw.strip().splitlines():
        if found_note:
            break
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().upper()
        val = val.strip()

        if key == "NEED_SEARCH":
            result["need_search"] = val.lower() in ("yes", "ya", "true", "1")
        elif key == "SEARCH_QUERY":
            result["search_query"] = val.strip('"').strip("'")
        elif key == "RECALL_TOPIC":
            # Normalisasi: kosong jika "kosong" atau "-"
            clean_val = val.strip('"').strip("'")
            result["recall_topic"] = "" if clean_val.lower() in ("kosong", "-", "") else clean_val
        elif key == "USER_EMOTION":
            emo = val.lower().strip()
            if emo in ("netral", "sedih", "cemas", "marah", "senang", "romantis"):
                result["user_emotion"] = emo
        elif key == "EMOTION_CONFIDENCE":
            conf = val.lower().strip()
            if conf in ("rendah", "sedang", "tinggi"):
                result["emotion_confidence"] = conf
        elif key == "TONE":
            result["tone"] = val.lower()
        elif key == "NOTE":
            result["note"] = val.strip('"').strip("'")
            found_note = True

    return result


def build_augmented_system(
    base_system: str,
    thought: dict,
    memory_context: str,
    web_result: str = "",
    emotion_guidance: str = "",
) -> str:
    parts = [base_system]

    if memory_context:
        parts.append(f"\n[Memori]\n{memory_context}")

    if web_result:
        if web_result.startswith("[INFO]"):
            parts.append(
                "\n[Instruksi Penting] Web search gagal. "
                "JANGAN mengarang data. Beritahu user dengan jujur bahwa "
                "kamu tidak bisa mengakses info terkini dan sarankan mereka "
                "cek sendiri di sumber terpercaya."
            )
        else:
            parts.append(
                f"\n[Hasil Web Search]\n{web_result[:600]}\n"
                "[Instruksi Penting] Gunakan informasi dari web search di atas "
                "sebagai dasar jawabanmu. Jangan mengabaikannya."
            )

    if emotion_guidance:
        parts.append(f"\n[Panduan Emosi]\n{emotion_guidance}")

    if thought.get("note"):
        parts.append(f"\n[Catatan]\n{thought['note']}")

    if thought.get("tone") == "informative":
        parts.append(
            "\n[Instruksi] Sampaikan informasi faktual dengan jelas, tetap gaya Asta."
        )

    return "".join(parts)


# ─── Helper ───────────────────────────────────────────────────────────────────

def extract_recent_context(conversation_history: list, n: int = 3) -> str:
    recent = conversation_history[-n:] if len(conversation_history) >= n else conversation_history
    lines = []
    for msg in recent:
        role = "Kamu" if msg["role"] == "user" else "Asta"
        content = msg["content"]
        if content:
            lines.append(f"{role}: {content[:120]}")
    return "\n".join(lines)


# ─── Debug Formatter ──────────────────────────────────────────────────────────

def format_thought_debug(thought: dict, web_result: str = "") -> str:
    lines = []
    lines.append("┌─ [Internal Thought - Raw Output] ────────────────────")
    raw = thought.get("raw", "").strip()
    if raw:
        for line in raw.splitlines():
            lines.append(f"│  {line}")
    else:
        lines.append("│  (kosong / gagal generate)")

    lines.append("├─ [Parsed] ────────────────────────────────────────────")
    lines.append(f"│  Search  : {'✓ akan search' if thought['need_search'] else '✗ tidak perlu'}")
    if thought["need_search"] and thought.get("search_query"):
        lines.append(f"│  Query   : {thought['search_query']}")
    recall = thought.get("recall_topic") or "–"
    lines.append(f"│  Recall  : {recall}")
    lines.append(f"│  Emotion : {thought.get('user_emotion', 'netral')} ({thought.get('emotion_confidence', 'rendah')})")
    lines.append(f"│  Tone    : {thought.get('tone', '–')}")
    lines.append(f"│  Note    : {thought.get('note') or '–'}")

    if thought["need_search"]:
        lines.append("├─ [Web Search Result] ─────────────────────────────────")
        if web_result and not web_result.startswith("[INFO]"):
            preview = web_result[:600]
            for line in preview.splitlines():
                if line.strip():
                    lines.append(f"│  {line}")
            if len(web_result) > 600:
                lines.append(f"│  ... ({len(web_result)} chars total)")
        elif web_result.startswith("[INFO]"):
            lines.append("│  ✗ Fetch gagal — model diperintahkan jujur ke user")
        else:
            lines.append("│  ✗ Tidak ada hasil")

    lines.append("└───────────────────────────────────────────────────────")
    return "\n".join(lines)
