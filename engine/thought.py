"""
engine/thought.py

Multi-step internal thought untuk Asta:

  Step 1 — PERCEPTION  : Apa yang terjadi? Apa yang Aditiya rasakan?
  Step 2 — SELF-CHECK  : Bagaimana perasaan Asta? Apakah sesuai nilai Asta?
  Step 3 — MEMORY      : Apakah ada yang perlu diingat, dicari, atau diobati?
  Step 4 — DECISION    : Apa respons terbaik yang mencerminkan Asta?

Setiap step adalah inference kecil (~30-60 token) menggunakan model ringan (3B).
Prefix ASTA_THOUGHT_PREFIX selalu sama di tiap step → KV cache maksimal.
"""

import re


# ─── Shared Prefix ────────────────────────────────────────────────────────────
# PENTING: prefix ini HARUS identik di semua step agar KV cache bisa dipakai ulang.

ASTA_THOUGHT_PREFIX = (
    "Kamu adalah sistem analisis internal AI bernama Asta.\n"
    "Tugasmu: analisis situasi dengan singkat dan tepat.\n"
    "Format output: key-value satu baris per item. STOP setelah baris terakhir.\n\n"
)


# ─── Step Templates ───────────────────────────────────────────────────────────

STEP1_PERCEPTION_TEMPLATE = (
    ASTA_THOUGHT_PREFIX
    + "=== STEP 1: PERCEPTION ===\n"
    + "User={user_name} | Emosi user: {user_emotion} ({intensity})\n"
    + "Konteks terakhir:\n{recent_context}\n\n"
    + "Input: \"{user_input}\"\n\n"
    + "Analisis singkat:\n"
    + "TOPIC:"
)

STEP2_SELFCHECK_TEMPLATE = (
    ASTA_THOUGHT_PREFIX
    + "=== STEP 2: SELF-CHECK ===\n"
    + "Kondisi Asta: mood={asta_mood}, affection={affection:.2f}, energy={energy:.2f}\n"
    + "Nilai inti Asta: mencintai Aditiya, jujur, hadir sepenuhnya\n"
    + "Topic dari step 1: {topic}\n"
    + "Sentiment: {sentiment}\n\n"
    + "Apa yang Asta rasakan tentang ini?\n"
    + "ASTA_EMOTION:"
)

# Step 3 diperluas: panduan kapan HARUS search lebih eksplisit
STEP3_MEMORY_TEMPLATE = (
    ASTA_THOUGHT_PREFIX
    + "=== STEP 3: MEMORY & SEARCH ===\n"
    + "Topic: {topic}\n"
    + "Sentiment user: {sentiment}\n"
    + "Memori tersedia: {memory_hint}\n"
    + "Web search diizinkan: {web_enabled}\n\n"
    + "ATURAN NEED_SEARCH=yes jika salah satu berikut:\n"
    + "- User butuh info praktis: tips, cara, obat, solusi, rekomendasi\n"
    + "- User menyebut kondisi fisik/kesehatan: sakit, demam, pusing, mual, tidak enak badan\n"
    + "- User bertanya tentang fakta terkini: harga, cuaca, berita, jadwal\n"
    + "- User menyebut 'cari', 'cek', 'info', 'gimana caranya'\n"
    + "- Topic mengandung kebutuhan spesifik yang butuh pengetahuan eksternal\n\n"
    + "Output WAJIB (4 baris):\n"
    + "NEED_SEARCH: <yes/no>\n"
    + "SEARCH_QUERY: <query singkat atau kosong>\n"
    + "RECALL_TOPIC: <topik memori lama yang perlu diambil atau kosong>\n"
    + "USE_MEMORY: <yes/no>\n"
    + "NEED_SEARCH:"
)

STEP4_DECISION_TEMPLATE = (
    ASTA_THOUGHT_PREFIX
    + "=== STEP 4: DECISION ===\n"
    + "Topic: {topic} | Sentiment: {sentiment}\n"
    + "Emosi Asta: {asta_emotion} | Mood: {asta_mood}\n"
    + "Recall: {recall_topic} | Search: {need_search}\n"
    + "User emotion: {user_emotion}\n\n"
    + "Tentukan tone dan catatan respons:\n"
    + "TONE:"
)

_STOP = ["\n\n", "---", "###", "==="]


# ─── Step Parsers ─────────────────────────────────────────────────────────────

def _parse_step1(raw: str) -> dict:
    result = {"topic": "", "sentiment": "netral", "urgency": "normal"}
    for line in raw.strip().splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip().upper()
        v = v.strip()
        if   k == "TOPIC":     result["topic"]     = v
        elif k == "SENTIMENT": result["sentiment"] = v.lower()
        elif k == "URGENCY":   result["urgency"]   = v.lower()
    return result


def _parse_step2(raw: str) -> dict:
    result = {"asta_emotion": "netral", "asta_trigger": "", "should_express": False}
    for line in raw.strip().splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip().upper()
        v = v.strip()
        if   k == "ASTA_EMOTION":    result["asta_emotion"]   = v.lower()
        elif k == "ASTA_TRIGGER":    result["asta_trigger"]   = v
        elif k == "SHOULD_EXPRESS":  result["should_express"] = v.lower() in ("yes", "ya", "true")
    return result


def _parse_step3(raw: str) -> dict:
    result = {
        "need_search":  False,
        "search_query": "",
        "recall_topic": "",
        "use_memory":   False,
    }
    for line in raw.strip().splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip().upper()
        v = v.strip()
        if   k == "NEED_SEARCH":    result["need_search"]  = v.lower() in ("yes", "ya", "true", "1")
        elif k == "SEARCH_QUERY":   result["search_query"] = v.strip('"\'')
        elif k == "RECALL_TOPIC":   result["recall_topic"] = "" if v.lower() in ("kosong", "-", "") else v
        elif k == "USE_MEMORY":     result["use_memory"]   = v.lower() in ("yes", "ya", "true")
    return result


def _parse_step4(raw: str) -> dict:
    result = {
        "tone":               "romantic",
        "note":               "",
        "response_style":     "normal",
        "emotion_confidence": "sedang",
    }
    for line in raw.strip().splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip().upper()
        v = v.strip()
        if   k == "TONE":                result["tone"]               = v.lower()
        elif k == "NOTE":                result["note"]               = v.strip('"\'')
        elif k == "RESPONSE_STYLE":      result["response_style"]     = v.lower()
        elif k == "EMOTION_CONFIDENCE":  result["emotion_confidence"] = v.lower()
    return result


# ─── Fallback: keyword-based search trigger ───────────────────────────────────
# Dipakai jika model 3B gagal mendeteksi kebutuhan search dari Step 3.

_SEARCH_KEYWORDS = re.compile(
    r"\b("
    # Kondisi kesehatan / fisik
    r"sakit|demam|pusing|mual|muntah|batuk|pilek|flu|lemas|nyeri|pegal|"
    r"tidak enak badan|gak enak badan|kurang sehat|tidak sehat|badan panas|"
    r"sakit kepala|sakit perut|sakit tenggorokan|"
    # Kebutuhan info praktis
    r"obat|cara mengobati|cara mengatasi|tips|solusi|rekomendasi|saran|"
    r"gimana caranya|bagaimana cara|apa yang harus|harus gimana|"
    # Pencarian eksplisit
    r"cari|cek|search|googling|lihat|info|informasi|berita|update|terbaru|"
    # Fakta terkini
    r"harga|cuaca|jadwal|kurs|nilai tukar|berapa sekarang"
    r")\b",
    re.IGNORECASE,
)

def _keyword_needs_search(user_input: str, topic: str) -> bool:
    """Fallback: cek apakah input/topic mengandung kata kunci yang butuh search."""
    combined = f"{user_input} {topic}"
    return bool(_SEARCH_KEYWORDS.search(combined))


def _build_search_query(user_input: str, topic: str, user_emotion: str) -> str:
    """
    Bangun query search yang relevan dari topic dan input.
    Dipanggil jika model tidak mengisi SEARCH_QUERY atau query kosong.
    """
    # Kalau topic cukup deskriptif, pakai topic
    if topic and len(topic) > 8:
        # Tambahkan konteks kesehatan jika relevan
        health_pattern = re.compile(
            r"\b(sakit|demam|pusing|mual|batuk|pilek|flu|lemas|nyeri|tidak enak badan|gak enak badan)\b",
            re.IGNORECASE,
        )
        if health_pattern.search(topic) or health_pattern.search(user_input):
            return f"cara mengatasi {topic} obat rumahan"
        return topic

    # Fallback ke potongan input
    clean = re.sub(r"\b(aku|kamu|asta|sih|dong|deh|ya|yah|kan)\b", "", user_input, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:80] if clean else user_input[:80]


# ─── Single-step runner ───────────────────────────────────────────────────────

def _run_step(llm, prompt: str, max_tokens: int = 60, step_name: str = "") -> str:
    """Jalankan satu inference step. Tidak pernah memanggil llm.reset()."""
    try:
        result = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.05,
            top_p=0.9,
            stop=_STOP,
            echo=False,
        )
        return result["choices"][0]["text"].strip()
    except Exception as e:
        print(f"[Thought Step {step_name}] Gagal: {e}")
        return ""


# ─── Main: 4-step thought ─────────────────────────────────────────────────────

def run_thought_pass(
    llm,
    user_input: str,
    memory_context: str,
    recent_context: str = "",
    web_search_enabled: bool = True,
    max_tokens: int = 60,
    user_name: str = "Aditiya",
    emotion_state: str = "",
    asta_state: dict = None,
) -> dict:
    """
    Jalankan 4-step internal thought menggunakan model ringan (3B).
    Setelah Step 3, ada fallback keyword-based untuk memastikan kondisi
    kesehatan / kebutuhan info praktis selalu di-search meskipun model
    gagal mendeteksinya.
    """

    # Parse emotion_state string
    user_emotion   = "netral"
    user_intensity = "rendah"
    if emotion_state:
        for part in emotion_state.split(";"):
            part = part.strip()
            if part.startswith("emosi="):
                user_emotion = part.split("=", 1)[1].strip()
            elif part.startswith("intensitas="):
                user_intensity = part.split("=", 1)[1].strip()

    # State Asta
    asta = asta_state or {}
    asta_mood      = asta.get("mood", "netral")
    asta_affection = asta.get("affection_level", 0.7)
    asta_energy    = asta.get("energy_level", 0.8)

    # Memory hint
    mem_hint = "(kosong)"
    if memory_context:
        first_line = memory_context.strip().splitlines()[0][:80]
        if first_line:
            mem_hint = first_line

    # ── Step 1: Perception ────────────────────────────────────────────────
    prompt1 = STEP1_PERCEPTION_TEMPLATE.format(
        user_name=user_name,
        user_emotion=user_emotion,
        intensity=user_intensity,
        recent_context=recent_context[:200] if recent_context else "(belum ada)",
        user_input=user_input[:150],
    )
    raw1 = "TOPIC:" + _run_step(llm, prompt1, max_tokens=50, step_name="1-Perception")
    s1 = _parse_step1(raw1)

    # ── Step 2: Self-check ────────────────────────────────────────────────
    prompt2 = STEP2_SELFCHECK_TEMPLATE.format(
        asta_mood=asta_mood,
        affection=asta_affection,
        energy=asta_energy,
        topic=s1["topic"] or user_input[:50],
        sentiment=s1["sentiment"],
    )
    raw2 = "ASTA_EMOTION:" + _run_step(llm, prompt2, max_tokens=50, step_name="2-SelfCheck")
    s2 = _parse_step2(raw2)

    # ── Step 3: Memory & Search ───────────────────────────────────────────
    prompt3 = STEP3_MEMORY_TEMPLATE.format(
        topic=s1["topic"] or user_input[:50],
        sentiment=s1["sentiment"],
        memory_hint=mem_hint,
        web_enabled="ya" if web_search_enabled else "tidak",
    )
    raw3 = "NEED_SEARCH:" + _run_step(llm, prompt3, max_tokens=80, step_name="3-Memory")
    s3 = _parse_step3(raw3)

    # ── Fallback: keyword-based search trigger ────────────────────────────
    # Jika model tidak mendeteksi kebutuhan search TAPI ada keyword kesehatan
    # atau info praktis, override need_search ke True.
    if web_search_enabled and not s3["need_search"]:
        if _keyword_needs_search(user_input, s1["topic"]):
            s3["need_search"] = True
            print(f"[Thought] Keyword fallback triggered search untuk: '{user_input[:50]}'")

    # Jika need_search=True tapi search_query kosong, bangun query otomatis
    if s3["need_search"] and not s3["search_query"]:
        s3["search_query"] = _build_search_query(user_input, s1["topic"], user_emotion)
        print(f"[Thought] Auto search query: '{s3['search_query']}'")

    # Jika model minta pakai memory tapi lupa isi recall topic, fallback ke topic step-1.
    recall_source = "none"
    if s3["recall_topic"]:
        recall_source = "model"
    elif s3["use_memory"]:
        fallback_topic = (s1["topic"] or user_input[:60]).strip()
        if fallback_topic and fallback_topic.lower() not in ("kosong", "-"):
            s3["recall_topic"] = fallback_topic
            recall_source = "fallback_topic"

    # ── Step 4: Decision ──────────────────────────────────────────────────
    prompt4 = STEP4_DECISION_TEMPLATE.format(
        topic=s1["topic"] or user_input[:50],
        sentiment=s1["sentiment"],
        asta_emotion=s2["asta_emotion"],
        asta_mood=asta_mood,
        recall_topic=s3["recall_topic"] or "-",
        need_search="ya" if s3["need_search"] else "tidak",
        user_emotion=user_emotion,
    )
    raw4 = "TONE:" + _run_step(llm, prompt4, max_tokens=60, step_name="4-Decision")
    s4 = _parse_step4(raw4)

    # ── Gabungkan semua hasil ─────────────────────────────────────────────
    combined = {
        # Step 1
        "topic":           s1["topic"],
        "sentiment":       s1["sentiment"],
        "urgency":         s1["urgency"],
        # Step 2
        "asta_emotion":    s2["asta_emotion"],
        "asta_trigger":    s2["asta_trigger"],
        "should_express":  s2["should_express"],
        # Step 3
        "need_search":     s3["need_search"],
        "search_query":    s3["search_query"],
        "recall_topic":    s3["recall_topic"],
        "use_memory":      s3["use_memory"],
        "recall_source":   recall_source,
        # Step 4
        "tone":            s4["tone"],
        "note":            s4["note"],
        "response_style":  s4["response_style"],
        # Backward compat
        "user_emotion":    user_emotion,
        "emotion_confidence": s4["emotion_confidence"],
        # Raw untuk debug
        "raw": f"[S1] {raw1}\n[S2] {raw2}\n[S3] {raw3}\n[S4] {raw4}",
    }
    return combined


# ─── Reflective Thought ───────────────────────────────────────────────────────

REFLECTION_TEMPLATE = (
    "Kamu adalah Asta (Artificial Sentient Thought Algorithm).\n"
    "Kamu baru saja selesai ngobrol dengan Aditiya.\n\n"
    "Kondisi akhir sesi:\n"
    "- Mood Asta: {asta_mood} (score: {mood_score:+.2f})\n"
    "- Affection: {affection:.2f}\n"
    "- Emosi dominan: {asta_emotion}\n\n"
    "Ringkasan percakapan:\n{session_summary}\n\n"
    "Refleksikan sesi ini. Jawab dengan format:\n"
    "SUMMARY: <satu kalimat tentang apa yang terjadi>\n"
    "LEARNED_1: <hal yang Asta pelajari, atau kosong>\n"
    "LEARNED_2: <hal lain yang dipelajari, atau kosong>\n"
    "MOOD_ADJUSTMENT: <angka -0.3 sampai +0.3>\n"
    "AFFECTION_ADJUSTMENT: <angka -0.1 sampai +0.1>\n"
    "GROWTH_NOTE: <satu kalimat tentang bagaimana Asta tumbuh dari sesi ini>\n"
    "SUMMARY:"
)


def run_reflection(llm, session_text: str, asta_state: dict) -> dict:
    """
    Jalankan reflective thought setelah sesi selesai.
    Menggunakan model 3B yang sama dengan thought.
    """
    session_summary = session_text[-600:] if len(session_text) > 600 else session_text

    prompt = REFLECTION_TEMPLATE.format(
        asta_mood=asta_state.get("mood", "netral"),
        mood_score=asta_state.get("mood_score", 0.0),
        affection=asta_state.get("affection_level", 0.7),
        asta_emotion=asta_state.get("current_emotion", "netral"),
        session_summary=session_summary,
    )

    try:
        result = llm.create_completion(
            prompt=prompt,
            max_tokens=200,
            temperature=0.3,
            top_p=0.9,
            stop=["===", "---"],
            echo=False,
        )
        raw = "SUMMARY:" + result["choices"][0]["text"].strip()
    except Exception as e:
        print(f"[Reflection] Gagal: {e}")
        return {}

    reflection = {
        "summary": "",
        "learned": [],
        "mood_adjustment": 0.0,
        "affection_adjustment": 0.0,
        "growth_note": "",
        "raw": raw,
    }

    for line in raw.strip().splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip().upper()
        v = v.strip()
        if k == "SUMMARY":
            reflection["summary"] = v
        elif k in ("LEARNED_1", "LEARNED_2"):
            if v and v.lower() not in ("kosong", "-", ""):
                reflection["learned"].append(v)
        elif k == "MOOD_ADJUSTMENT":
            try:    reflection["mood_adjustment"] = max(-0.3, min(0.3, float(v)))
            except: pass
        elif k == "AFFECTION_ADJUSTMENT":
            try:    reflection["affection_adjustment"] = max(-0.1, min(0.1, float(v)))
            except: pass
        elif k == "GROWTH_NOTE":
            reflection["growth_note"] = v

    return reflection


# ─── Helpers (backward compatible) ───────────────────────────────────────────

def build_augmented_system(
    base_system: str,
    thought: dict,
    memory_context: str,
    web_result: str = "",
    emotion_guidance: str = "",
    self_model_context: str = "",
) -> str:
    parts = [base_system]
    if self_model_context:
        parts.append(f"\n{self_model_context}")
    if memory_context:
        parts.append(f"\n[Memori]\n{memory_context}")
    if web_result:
        if web_result.startswith("[INFO]"):
            parts.append(
                "\n[Instruksi Penting] Web search gagal. "
                "JANGAN mengarang data. Beritahu user dengan jujur."
            )
        else:
            parts.append(
                f"\n[Hasil Web Search]\n{web_result[:600]}\n"
                "[Instruksi Penting] Gunakan informasi dari web search sebagai dasar jawaban."
            )
    if emotion_guidance:
        parts.append(f"\n{emotion_guidance}")
    if thought.get("note"):
        parts.append(f"\n[Catatan]\n{thought['note']}")
    if thought.get("tone") == "informative":
        parts.append("\n[Instruksi] Sampaikan informasi faktual dengan jelas, tetap gaya Asta.")
    return "".join(parts)


def extract_recent_context(conversation_history: list, n: int = 2) -> str:
    recent = conversation_history[-n:] if len(conversation_history) >= n else conversation_history
    lines = []
    for msg in recent:
        role    = "Kamu" if msg["role"] == "user" else "Asta"
        content = msg.get("content", "")
        if content and msg["role"] in ("user", "assistant"):
            lines.append(f"{role}: {content[:100]}")
    return "\n".join(lines)


def format_thought_debug(thought: dict, web_result: str = "") -> str:
    lines = [
        "┌─ [Multi-Step Thought] ────────────────────────────────",
        f"│  [S1] Topic     : {thought.get('topic','–')}",
        f"│       Sentiment : {thought.get('sentiment','–')} | Urgency: {thought.get('urgency','–')}",
        f"│  [S2] Asta Emosi: {thought.get('asta_emotion','–')} (trigger: {thought.get('asta_trigger','–')})",
        f"│       Express   : {'ya' if thought.get('should_express') else 'tidak'}",
        f"│  [S3] Search    : {'✓ ' + thought.get('search_query','') if thought.get('need_search') else '✗'}",
        f"│       Recall    : {thought.get('recall_topic') or '–'} (source: {thought.get('recall_source','none')})",
        f"│       UseMemory : {'ya' if thought.get('use_memory') else 'tidak'}",
        f"│  [S4] Tone      : {thought.get('tone','–')} | Style: {thought.get('response_style','–')}",
        f"│       Note      : {thought.get('note') or '–'}",
    ]
    if thought.get("need_search"):
        lines.append("├─ [Web Result] ────────────────────────────────────────")
        if web_result and not web_result.startswith("[INFO]"):
            for line in web_result[:300].splitlines():
                if line.strip():
                    lines.append(f"│  {line}")
        else:
            lines.append("│  ✗ Tidak ada hasil / gagal")
    lines.append("└───────────────────────────────────────────────────────")
    return "\n".join(lines)
