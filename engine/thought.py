"""
engine/thought.py — Internal Thought Engine untuk Asta.

Fix v3:
  1. Thought pass menerima recent_context (2-3 pesan terakhir)
     agar query yang dihasilkan relevan dengan topik percakapan aktif,
     bukan mengarang dari memori secara acak.
  2. build_augmented_system memberi instruksi eksplisit ketika ada
     web_result agar model benar-benar menggunakannya, bukan mengabaikannya.
"""

from typing import Optional

# ─── Thought Prompt ────────────────────────────────────────────────────────────

_THOUGHT_SYSTEM = """\
Kamu adalah sistem analisis input. Baca konteks percakapan dan input terbaru,
lalu tulis SATU analisis.

ATURAN NEED_SEARCH:
- yes: user secara eksplisit meminta verifikasi, info faktual terkini, atau
       menyebut kata seperti "cari", "cek", "verifikasi", "berapa", "siapa",
       "cuaca", "harga", "kurs", "berita"
- no : sapaan, obrolan biasa, perasaan, pernyataan umum, pertanyaan retoris

Jika NEED_SEARCH yes, buat SEARCH_QUERY yang spesifik berdasarkan
TOPIK PERCAKAPAN TERKINI (bukan dari memori atau asumsi sendiri).

FORMAT WAJIB (berhenti tepat setelah NOTE, tidak ada teks lain):
NEED_SEARCH: yes/no
SEARCH_QUERY: <query spesifik sesuai topik percakapan, kosong jika no>
RECALL_TOPIC: <topik memori relevan, kosong jika tidak ada>
TONE: romantic/casual/informative
NOTE: <catatan singkat maks 10 kata>\
"""

_STOP_TOKENS = ["\n\n", "---", "Input pengguna:", "Analisis:", "</thought>", "###"]


def run_thought_pass(
    llm,
    user_input: str,
    memory_context: str,
    recent_context: str = "",   # ← BARU: 2-3 pesan terakhir sebagai konteks
    web_search_enabled: bool = True,
    max_tokens: int = 80,
) -> dict:
    """
    Model memutuskan apakah perlu search berdasarkan:
    - recent_context: topik percakapan yang sedang berlangsung
    - user_input: input terbaru user
    Bukan dari memori atau asumsi sendiri.
    """
    mem_hint = ""
    if memory_context:
        first_line = memory_context.strip().splitlines()[0]
        mem_hint = first_line[:100]

    # recent_context memberikan topik aktif percakapan ke thought pass
    context_block = ""
    if recent_context:
        context_block = f"Konteks percakapan terkini:\n{recent_context}\n\n"

    prompt = (
        f"{_THOUGHT_SYSTEM}\n\n"
        f"Hint memori: {mem_hint or '(kosong)'}\n"
        f"Web search: {'tersedia' if web_search_enabled else 'tidak tersedia'}\n\n"
        f"{context_block}"
        f"Input terbaru: \"{user_input}\"\n\n"
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
    """Parse output thought — berhenti setelah NOTE pertama."""
    result = {
        "need_search": False,
        "search_query": "",
        "recall_topic": "",
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
            result["recall_topic"] = val
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
) -> str:
    """
    Bangun system prompt augmented untuk response pass.
    Jika ada web_result, beri instruksi eksplisit agar model benar-benar memakainya.
    """
    parts = [base_system]

    if memory_context:
        parts.append(f"\n[Memori]\n{memory_context}")

    if web_result:
        if web_result.startswith("[INFO]"):
            # Fetch gagal
            parts.append(
                "\n[Instruksi Penting] Web search gagal. "
                "JANGAN mengarang data. Beritahu user dengan jujur bahwa "
                "kamu tidak bisa mengakses info terkini dan sarankan mereka "
                "cek sendiri di sumber terpercaya."
            )
        else:
            # Ada hasil — instruksikan model untuk MENGGUNAKAN hasil ini
            parts.append(
                f"\n[Hasil Web Search]\n{web_result[:600]}\n"
                "[Instruksi Penting] Gunakan informasi dari web search di atas "
                "sebagai dasar jawabanmu. Jangan mengabaikannya atau mengganti "
                "dengan pengetahuan sendiri. Sebutkan sumber jika relevan."
            )

    if thought.get("note"):
        parts.append(f"\n[Catatan]\n{thought['note']}")

    if thought.get("tone") == "informative":
        parts.append(
            "\n[Instruksi] Sampaikan informasi faktual dengan jelas, tetap gaya Asta."
        )

    return "".join(parts)


# ─── Helper: Ekstrak Recent Context ───────────────────────────────────────────

def extract_recent_context(conversation_history: list, n: int = 3) -> str:
    """
    Ambil N pesan terakhir dari conversation_history sebagai konteks topik aktif.
    Dipakai untuk memberi thought pass konteks percakapan yang sedang berlangsung.
    """
    recent = conversation_history[-n:] if len(conversation_history) >= n else conversation_history
    lines = []
    for msg in recent:
        role = "Kamu" if msg["role"] == "user" else "Asta"
        content = msg["content"]
        if content:
            # Potong jika terlalu panjang
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
    lines.append(f"│  Recall  : {thought.get('recall_topic') or '–'}")
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
