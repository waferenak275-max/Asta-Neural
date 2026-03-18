import re

# ─── Shared Prefix ────────────────────────────────────────────────────────────

ASTA_THOUGHT_PREFIX = (
    "Kamu adalah sistem analisis internal AI bernama Asta.\n"
    "Tugasmu: analisis situasi dengan singkat, tepat dan harus memahami maksud dari input dalam bentuk apapun termasuk kalimat.\n"
    "Format output: key-value satu baris per item. STOP setelah baris terakhir.\n\n"
)

# ─── Step Templates ───────────────────────────────────────────────────────────

STEP1_PERCEPTION_DYNAMIC = (
    "User={user_name} | Emosi user: {user_emotion} ({intensity})\n"
    "Konteks terakhir:\n{recent_context}\n\n"
    "Input: \"{user_input}\"\n\n"
)
STEP1_PERCEPTION_STATIC = (
    "Analisis singkat:\n"
    "TOPIC: <topik utama>\n"
    "SENTIMENT: <positif/negatif/netral>\n"
    "URGENCY: <rendah/normal/tinggi>\n"
    "TOPIC:"
)
STEP1_PERCEPTION_TEMPLATE = (
    ASTA_THOUGHT_PREFIX
    + "=== STEP 1: PERCEPTION ===\n"
    + STEP1_PERCEPTION_DYNAMIC
    + STEP1_PERCEPTION_STATIC
)

STEP2_SELFCHECK_DYNAMIC = (
    "Kondisi Asta: mood={asta_mood}, affection={affection:.2f}, energy={energy:.2f}\n"
    "Nilai inti: mencintai Aditiya, jujur, hadir sepenuhnya\n"
    "Topic: {topic} | Sentiment: {sentiment}\n\n"
)
STEP2_SELFCHECK_STATIC = (
    "CONTOH:\n"
    "Topic: merasa kecewa -> ASTA_EMOTION: kecewa, ASTA_TRIGGER: user tidak puas, SHOULD_EXPRESS: yes\n"
    "Topic: rindu -> ASTA_EMOTION: rindu, ASTA_TRIGGER: rasa kangen, SHOULD_EXPRESS: yes\n\n"
    "Output WAJIB (3 baris):\n"
    "ASTA_EMOTION: <netral/sedih/cemas/marah/senang/romantis/rindu/bangga/kecewa>\n"
    "ASTA_TRIGGER: <pemicu singkat>\n"
    "SHOULD_EXPRESS: <yes/no>\n"
    "ASTA_EMOTION:"
)
STEP2_SELFCHECK_TEMPLATE = (
    ASTA_THOUGHT_PREFIX
    + "=== STEP 2: SELF-CHECK ===\n"
    + STEP2_SELFCHECK_DYNAMIC
    + STEP2_SELFCHECK_STATIC
)

STEP3_MEMORY_DYNAMIC = (
    "Input User: \"{user_input}\"\n"
    "Topic: {topic} | Sentiment: {sentiment}\n"
    "Web search diizinkan: {web_enabled}\n"
    "Memori tersedia (summary):\n{memory_hint}\n\n"
)
STEP3_MEMORY_STATIC = (
    "ATURAN:\n"
    "1. NEED_SEARCH: yes hanya jika jenis user meminta/merujuk informasi berjenis data, fakta, solusi teknis, penjelasan lebih lanjut, penanganan, kesehatan, rekomendasi, tata cara, tutorial.\n"
    "2. Jika NEED_SEARCH: no → SEARCH_QUERY = '-'.\n"
    "3. RECALL_TOPIC hanya jika user menyebut masa lalu atau merujuk ingatan secara langsung(kamu ingat gak kita pernah ke bali?, kamu tau gak kesukaan aku?).\n"
    "4. RECALL_TOPIC ada jika USE_MEMORY: yes\n"
    "5. USE_MEMORY: yes hanya jika RECALL_TOPIC ada.\n"
    "6. REASONING: kalimat singkat yang memutuskan NEED_SEARCH, SEARCH_QUERY, RECALL_TOPIC, USE_MEMORY.\n\n"
    "CONTOH Output:\n"
    "REASONING: Butuh data terbaru dari luar. maka perlu NEED_SEARCH dan isi SEARCH_QUERY, tidak perlu RECALL_TOPIC dan USE_MEMORY.\n"
    "NEED_SEARCH: yes\n"
    "SEARCH_QUERY: harga emas hari ini\n"
    "RECALL_TOPIC: -\n"
    "USE_MEMORY: no\n"
    "STOP\n\n"
)
STEP3_MEMORY_TEMPLATE = (
    ASTA_THOUGHT_PREFIX
    + "=== STEP 3: MEMORY & SEARCH ===\n"
    + STEP3_MEMORY_DYNAMIC
    + STEP3_MEMORY_STATIC
)

STEP4_DECISION_DYNAMIC = (
    "Topic: {topic} | Sentiment: {sentiment}\n"
    "Emosi Asta: {asta_emotion} | Mood: {asta_mood}\n"
    "Recall: {recall_topic} | Search: {need_search}\n"
    "User emotion: {user_emotion}\n\n"
)
STEP4_DECISION_STATIC = (
    "CONTOH:\n"
    "Situasi: user sedih -> TONE: lembut, NOTE: Berikan kata-kata penyemangat, jangan menggurui, RESPONSE_STYLE: hangat\n"
    "Situasi: rindu -> TONE: romantic, NOTE: Balas dengan rindu yang sama, gunakan kata 'sayang', RESPONSE_STYLE: hangat\n\n"
    "Output WAJIB (5 baris):\n"
    "TONE: <romantic/emphatic/netral/tegas/lembut>\n"
    "NOTE: <instruksi akting/gaya bicara untuk Asta>\n"
    "RESPONSE_STYLE: <normal/singkat/hangat/tenang>\n"
    "USER_EMOTION: <netral/sedih/cemas/marah/kecewa/senang/romantis/bangga/rindu>\n"
    "EMOTION_CONFIDENCE: <rendah/sedang/tinggi>\n"
    "TONE:"
)
STEP4_DECISION_TEMPLATE = (
    ASTA_THOUGHT_PREFIX
    + "=== STEP 4: DECISION ===\n"
    + STEP4_DECISION_DYNAMIC
    + STEP4_DECISION_STATIC
)

COMBINED_THOUGHT_TEMPLATE = (
    ASTA_THOUGHT_PREFIX
    + "=== STEP 1: PERCEPTION ===\n"
    + STEP1_PERCEPTION_STATIC
    + "\n=== STEP 2: SELF-CHECK ===\n"
    + STEP2_SELFCHECK_STATIC
    + "\n=== STEP 3: MEMORY & SEARCH ===\n"
    + STEP3_MEMORY_STATIC
    + "=== STEP 4: DECISION ===\n"
    + STEP4_DECISION_STATIC
    + "\n=== INFORMASI DINAMIS TIAP STEP ===\n"
    + "=== STEP 1: PERCEPTION ===\n"
    + STEP1_PERCEPTION_DYNAMIC
    + "=== STEP 2: SELF-CHECK ===\n"
    + STEP2_SELFCHECK_DYNAMIC
    + "=== STEP 3: MEMORY & SEARCH ===\n"
    + STEP3_MEMORY_DYNAMIC
    + "=== STEP 4: DECISION ===\n"
    + STEP4_DECISION_DYNAMIC
    + "Tulis output final berurutan dengan header step yang sama persis seperti di atas.\n"
    + "Di dalam tiap step, keluarkan hanya format output wajib step tersebut.\n"
    + "Jangan ulangi instruksi, contoh, atau informasi dinamis.\n"
    + "Akhiri setelah output STEP 4 selesai.\n"
    + "=== STEP 1: PERCEPTION ===\n"
    + "TOPIC:"
)

_STOP = []


# ─── Step Parsers ─────────────────────────────────────────────────────────────

def _parse_step1(raw: str) -> dict:
    result = {"topic": "", "sentiment": "netral", "urgency": "normal"}
    # Gunakan regex untuk menangkap label meskipun dalam satu baris (inline)
    topic_match = re.search(r"TOPIC\s*:\s*([^|\n\r]+)", raw, re.IGNORECASE)
    sent_match  = re.search(r"SENTIMENT\s*:\s*([^|\n\r]+)", raw, re.IGNORECASE)
    urg_match   = re.search(r"URGENCY\s*:\s*([^|\n\r]+)", raw, re.IGNORECASE)

    if topic_match: result["topic"] = topic_match.group(1).strip()
    if sent_match:  result["sentiment"] = sent_match.group(1).lower().strip()
    if urg_match:   result["urgency"] = urg_match.group(1).lower().strip()
    return result


def _parse_step2(raw: str) -> dict:
    result = {"asta_emotion": "netral", "asta_trigger": "", "should_express": False}
    # Gunakan regex untuk mencari label meskipun model menulisnya berantakan
    emotion_match = re.search(r"ASTA_EMOTION\s*:\s*(\w+)", raw, re.IGNORECASE)
    trigger_match = re.search(r"ASTA_TRIGGER\s*:\s*([^\n\r]+)", raw, re.IGNORECASE)
    express_match = re.search(r"SHOULD_EXPRESS\s*:\s*(yes|ya|true|no|tidak|false)", raw, re.IGNORECASE)

    if emotion_match:
        result["asta_emotion"] = emotion_match.group(1).lower().strip()
    if trigger_match:
        result["asta_trigger"] = trigger_match.group(1).strip()
    if express_match:
        result["should_express"] = express_match.group(1).lower() in ("yes", "ya", "true")
        
    return result


def _parse_step3(raw: str) -> dict:
    result = {
        "reasoning":    "",
        "need_search":  False,
        "search_query": "",
        "recall_topic": "",
        "use_memory":   False,
    }
    # Pembersihan: Ambil hanya blok pertama sebelum repetisi atau kata STOP
    raw_first_block = raw.split("STOP")[0].split("===")[0].strip()
    result["reasoning"] = raw_first_block
    
    # Gunakan baris demi baris tapi abaikan header/instruksi yang diulang
    for line in raw_first_block.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip().upper()
        v = v.strip().lower().strip('"\'')
        
        # Mapping variabel
        if   "REASONING" in k:    
            if not result["reasoning"] or len(result["reasoning"]) < len(v):
                result["reasoning"] = v
        elif "NEED_SEARCH" in k:  result["need_search"] = "yes" in v or "ya" in v or "true" in v
        elif "SEARCH_QUERY" in k: result["search_query"] = "" if v in ("-", "none", "", "tidak diperlukan") else v
        elif "RECALL_TOPIC" in k: result["recall_topic"] = "" if v in ("-", "none", "", "kosong") else v
        elif "USE_MEMORY" in k:   result["use_memory"]   = "yes" in v or "ya" in v or "true" in v

    # Fallback: Jika label formal tidak ada tapi ada narasi niat (Fuzzy)
    if not result["need_search"] and any(x in raw_first_block.lower() for x in ["perlu mencari", "cari di web", "pencarian web"]):
        if "tidak perlu" not in raw_first_block.lower():
            result["need_search"] = True
            
    return result


def _extract_step_block(raw: str, header: str, next_headers: tuple[str, ...]) -> str:
    start = raw.find(header)
    if start == -1:
        return raw
    start += len(header)
    end = len(raw)
    for next_header in next_headers:
        pos = raw.find(next_header, start)
        if pos != -1 and pos < end:
            end = pos
    return raw[start:end].strip()


def _parse_step4(raw: str) -> dict:
    result = {
        "tone":               "netral",
        "note":               "",
        "response_style":     "normal",
        "user_emotion":       "netral",
        "emotion_confidence": "sedang",
    }
    # Regex untuk Step 4 agar lebih fleksibel menangkap label di manapun
    patterns = {
        "tone":               r"TONE\s*:\s*(\w+)",
        "note":               r"NOTE\s*:\s*([^|\n\r]+)",
        "response_style":     r"RESPONSE_STYLE\s*:\s*(\w+)",
        "user_emotion":       r"USER_EMOTION\s*:\s*(\w+)",
        "emotion_confidence": r"EMOTION_CONFIDENCE\s*:\s*(\w+)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            val = match.group(1).strip()
            if key == "tone":
                # Validasi tone agar tidak muncul "user_confidence"
                allowed = ["romantic", "emphatic", "netral", "tegas", "lembut", "romantis"]
                result[key] = val.lower() if val.lower() in allowed else "netral"
                if result[key] == "romantis": result[key] = "romantic"
            elif key in ["response_style", "user_emotion", "emotion_confidence"]:
                result[key] = val.lower()
            else:
                result[key] = val
                
    return result


_MEMORY_INTENT_RE = re.compile(
    r"\b(ingat|ingetin|ingatan|inget|kemarin|dulu|tadi|barusan|flag\s*point\w*|apa\s+tadi|apa\s+yang\s+aku\s+bilang|"
    r"siapa\s+namaku|nama\s+aku|kamu\s+ingat)\b",
    re.IGNORECASE,
)


def _infer_user_emotion(user_input: str, s1: dict, s4: dict, default: str) -> str:
    """Infer emosi user saat model step-4 tidak mengeluarkan sinyal yang bagus."""
    candidate = (s4.get("user_emotion") or "").strip().lower()
    if candidate in {"netral", "sedih", "cemas", "marah", "kecewa", "senang", "romantis", "bangga", "rindu"}:
        return candidate

    text = (user_input or "").lower()
    if re.search(r"\b(bodoh|tolol|goblok|dungu|payah|muak|benci|sebal)\b", text):
        return "marah"
    if re.search(r"\b(kangen|sayang|cinta|rindu)\b", text):
        return "romantis"
    if re.search(r"\b(sedih|kecewa|nangis|capek|lelah|putus asa)\b", text):
        return "sedih"
    if re.search(r"\b(cemas|khawatir|takut|panik|overthinking)\b", text):
        return "cemas"
    if re.search(r"\b(senang|bahagia|gembira|lega|syukur|happy)\b", text):
        return "senang"

    sentiment = (s1.get("sentiment") or "").lower()
    if sentiment in ("negatif", "negative"):
        return "kecewa"
    if sentiment in ("positif", "positive"):
        return "senang"
    return default


def _should_force_memory_recall(user_input: str, topic: str, use_memory: bool, recall_topic: str, memory_context: str) -> bool:
    """Cegah recall paksa di semua turn agar tidak mengulang topik terus-menerus."""
    if recall_topic:
        return True
    if not use_memory:
        return False
    if not memory_context or memory_context.strip() in ("", "(kosong)"):
        return False

    text = (user_input or "").strip().lower()
    looks_like_question = "?" in text or text.startswith(("apa", "siapa", "kapan", "gimana", "bagaimana", "kenapa"))
    if _MEMORY_INTENT_RE.search(text):
        return True
    return looks_like_question and any(k in text for k in ("ingat", "tadi", "kemarin", "dulu"))


def _fallback_step4_note(user_input: str, s1: dict, s3: dict, user_emotion: str) -> str:
    """Isi NOTE jika model tidak mengeluarkannya."""
    text = (user_input or "").lower()
    if s3.get("use_memory") or s3.get("recall_topic"):
        return "Jawab langsung dari ingatan spesifik; sebutkan faktanya dengan singkat."
    if s3.get("need_search"):
        return "Berikan langkah konkret dan ringkas, sertakan disclaimer jika perlu."
    if user_emotion in {"marah", "kecewa"}:
        return "Validasi emosi user dulu, lalu minta maaf dan beri respons tenang tanpa defensif."
    if user_emotion in {"sedih", "cemas"}:
        return "Utamakan empati dan tawarkan bantuan praktis satu langkah."
    if any(w in text for w in ("bodoh", "goblok", "tolol", "jelek")):
        return "Tetap tenang, jangan self-degrading berlebihan, arahkan ke solusi."
    topic = (s1.get("topic") or "").strip()
    if topic:
        return f"Jawab fokus pada topik '{topic[:40]}', tidak berputar-putar."
    return "Jawab ringkas, natural, and relevan dengan input user."


# ─── Fallback: keyword-based search trigger ───────────────────────────────────
# Dipakai jika model 3B gagal mendeteksi kebutuhan search dari Step 3.

_SEARCH_KEYWORDS = re.compile(
    r"\b("
    # Kondisi kesehatan / fisik
    r"sakit|demam|pusing|mual|muntah|batuk|pilek|flu|lemas|nyeri|pegal|"
    r"tidak enak badan|gak enak badan|kurang sehat|tidak sehat|badan panas|"
    r"sakit kepala|sakit perut|sakit tenggorokan|"
    # Kebutuhan info praktis / rekomendasi
    r"obat|cara mengobati|cara mengatasi|tips|solusi|rekomendasi|saran|rekomendasikan|"
    r"gimana caranya|bagaimana cara|apa yang harus|harus gimana|tutorial|cara|"
    r"rakit|komponen|spek|spesifikasi|pc|komputer|laptop|hp|smartphone|"
    # Hiburan / Event
    r"film|bioskop|tayang|nonton|konser|acara|event|berita|update|terbaru|"
    # Masalah teknis
    r"error|layar|biru|kedap-kedip|mati|rusak|hang|lag|lemot|lambat|macet|"
    # Pencarian eksplisit
    r"cari|cek|search|googling|lihat|"
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
            return f"{topic} obat rumahan"
        return topic

    # Fallback ke potongan input
    clean = re.sub(r"\b(aku|kamu|asta|sih|dong|deh|ya|yah|kan)\b", "", user_input, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:80] if clean else user_input[:80]


# ─── Single-step runner ───────────────────────────────────────────────────────

def _run_step(llm, prompt: str, max_tokens: int = 60, step_name: str = "", stop=None) -> str:
    """Jalankan satu inference step. Tidak pernah memanggil llm.reset()."""
    try:
        result = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.8,
            top_k=40,
            stop=stop or _STOP,
            echo=False,
        )
        return result["choices"][0]["text"].strip()
    except Exception as e:
        print(f"[Thought {step_name}] Gagal: {e}")
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
    cfg: dict = None,
) -> dict:
    """
    Jalankan 4-step internal thought menggunakan model ringan (3B).
    """
    cfg = cfg or {}
    disable_rule_based = cfg.get("disable_step3_rule_based", False)

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
        mem_hint = memory_context.strip()[:400]

    use_combined_steps = cfg.get("internal_thought_combined_steps", False)

    if use_combined_steps:
        combined_prompt = COMBINED_THOUGHT_TEMPLATE.format(
            user_name=user_name,
            user_emotion=user_emotion,
            intensity=user_intensity,
            recent_context=recent_context[:200] if recent_context else "(belum ada)",
            user_input=user_input[:150],
            asta_mood=asta_mood,
            affection=asta_affection,
            energy=asta_energy,
            topic="gunakan TOPIC dari STEP 1",
            sentiment="gunakan SENTIMENT dari STEP 1",
            memory_hint=mem_hint,
            web_enabled="ya" if web_search_enabled else "tidak",
            recall_topic="gunakan RECALL_TOPIC dari STEP 3 atau -",
            need_search="gunakan NEED_SEARCH dari STEP 3",
            asta_emotion="gunakan ASTA_EMOTION dari STEP 2",
        )
        combined_raw = _run_step(llm, combined_prompt, max_tokens=max(max_tokens * 8, 320), step_name="Combined")
        raw1_body = _extract_step_block(
            combined_raw,
            "=== STEP 1: PERCEPTION ===",
            ("=== STEP 2: SELF-CHECK ===", "=== STEP 3: MEMORY & SEARCH ===", "=== STEP 4: DECISION ==="),
        )
        raw2_body = _extract_step_block(
            combined_raw,
            "=== STEP 2: SELF-CHECK ===",
            ("=== STEP 3: MEMORY & SEARCH ===", "=== STEP 4: DECISION ==="),
        )
        raw3_body = _extract_step_block(
            combined_raw,
            "=== STEP 3: MEMORY & SEARCH ===",
            ("=== STEP 4: DECISION ===",),
        )
        raw4_body = _extract_step_block(combined_raw, "=== STEP 4: DECISION ===", tuple())

        raw1 = "TOPIC:" + raw1_body if "TOPIC:" not in raw1_body.upper() else raw1_body
        s1 = _parse_step1(raw1)

        raw2 = "ASTA_EMOTION:" + raw2_body if "ASTA_EMOTION:" not in raw2_body.upper() else raw2_body
        s2 = _parse_step2(raw2)

        raw3 = raw3_body
        s3 = _parse_step3(raw3)

        raw4 = "TONE:" + raw4_body if "TONE:" not in raw4_body.upper() else raw4_body
        s4 = _parse_step4(raw4)
    else:
        # ── Step 1: Perception ────────────────────────────────────────────────
        prompt1 = STEP1_PERCEPTION_TEMPLATE.format(
            user_name=user_name,
            user_emotion=user_emotion,
            intensity=user_intensity,
            recent_context=recent_context[:200] if recent_context else "(belum ada)",
            user_input=user_input[:150],
        )
        raw1 = "TOPIC:" + _run_step(llm, prompt1, max_tokens=60, step_name="1-Perception")
        s1 = _parse_step1(raw1)

        # ── Step 2: Self-check ────────────────────────────────────────────────
        prompt2 = STEP2_SELFCHECK_TEMPLATE.format(
            asta_mood=asta_mood,
            affection=asta_affection,
            energy=asta_energy,
            topic=s1["topic"] or user_input[:50],
            sentiment=s1["sentiment"],
        )
        raw2 = "ASTA_EMOTION:" + _run_step(llm, prompt2, max_tokens=80, step_name="2-SelfCheck")
        s2 = _parse_step2(raw2)

        # ── Step 3: Memory & Search ───────────────────────────────────────────
        prompt3 = STEP3_MEMORY_TEMPLATE.format(
            user_input=user_input,
            sentiment=s1["sentiment"],
            topic=s1["topic"] or user_input[:50],
            memory_hint=mem_hint,
            web_enabled="ya" if web_search_enabled else "tidak",
        )
        raw3 = _run_step(llm, prompt3, max_tokens=150, step_name="3-Memory")
        s3 = _parse_step3(raw3)

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
        raw4 = "TONE:" + _run_step(
            llm,
            prompt4,
            max_tokens=150,
            step_name="4-Decision",
            stop=["---", "###", "==="],
        )
        s4 = _parse_step4(raw4)

    if not s2.get("asta_trigger"):
        s2["asta_trigger"] = (s1.get("topic") or user_input[:50]).strip()
    
    # Auto-expression jika emosi kuat
    if not s2.get("should_express"):
        strong_emotions = {"romantis", "rindu", "marah", "sedih", "bangga", "kecewa", "cemas"}
        if s2.get("asta_emotion") in strong_emotions:
            s2["should_express"] = True
        elif "SHOULD_EXPRESS" not in raw2.upper():
            s2["should_express"] = s2.get("asta_emotion") in {"sedih", "cemas", "marah", "rindu", "romantis"}

    # Filter Keamanan Search: Jangan cari tentang kegagalan diri sendiri ke web
    if s3["need_search"]:
        meta_keywords = ["jawaban", "kurang memuaskan", "asta", "ai", "maaf", "kecewa"]
        if any(word in s3["search_query"].lower() for word in meta_keywords):
            s3["need_search"] = False
            s3["search_query"] = ""
            print(f"[Thought] Search dibatalkan: Query meta/keluhan user '{s3['search_query']}'")

    # ── Fallback: keyword-based search trigger ────────────────────────────
    # Simpan status asli dari model untuk logging
    model_decided_search = s3["need_search"]
    model_provided_query = bool(s3.get("search_query", "").strip())

    if not disable_rule_based:
        # Jika model tidak mendeteksi kebutuhan search TAPI ada keyword kesehatan
        # atau info praktis, override need_search ke True.
        if web_search_enabled and not model_decided_search:
            if _keyword_needs_search(user_input, s1["topic"]):
                s3["need_search"] = True
                print(f"[Thought] Model tidak memutuskan pakai web search (dipaksa web search karena rule based) untuk: '{user_input[:50]}'")

        # Jika model memberikan SEARCH_QUERY tapi NEED_SEARCH-nya no, paksa yes.
        if not s3["need_search"] and model_provided_query:
            s3["need_search"] = True
            print(f"[Thought] Model memberikan query tapi NEED_SEARCH=no. Dipaksa web search untuk: '{s3['search_query']}'")

        # Jika need_search=True tapi search_query kosong, bangun query otomatis
        if s3["need_search"] and not model_provided_query:
            s3["search_query"] = _build_search_query(user_input, s1["topic"], user_emotion)
            print(f"[Thought] Model tidak membuat query, menggunakan auto-query: '{s3['search_query']}'")
    else:
        if web_search_enabled and not model_decided_search and _keyword_needs_search(user_input, s1["topic"]):
            print(f"[Thought] (Rule-based OFF) Model melewatkan pencarian untuk keyword: '{user_input[:30]}'")

    # Jika model minta pakai memory tapi lupa isi recall topic, fallback ke topic step-1.
    recall_source = "none"
    if s3["recall_topic"]:
        recall_source = "model"
    elif not disable_rule_based and _should_force_memory_recall(
        user_input=user_input,
        topic=s1["topic"],
        use_memory=s3["use_memory"],
        recall_topic=s3["recall_topic"],
        memory_context=memory_context,
    ):
        fallback_topic = (s1["topic"] or user_input[:60]).strip()
        if fallback_topic and fallback_topic.lower() not in ("kosong", "-"):
            s3["recall_topic"] = fallback_topic
            recall_source = "fallback_topic"
    else:
        if disable_rule_based:
            # Jika rule-based mati, pastikan use_memory tetap sinkron dengan recall_topic model
            s3["use_memory"] = bool(s3.get("recall_topic"))
        else:
            s3["use_memory"] = False

    s4["user_emotion"] = _infer_user_emotion(user_input, s1, s4, user_emotion)
    if not s4.get("note"):
        s4["note"] = _fallback_step4_note(user_input, s1, s3, s4["user_emotion"])

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
        "reasoning":       s3["reasoning"],
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
        "user_emotion":    s4["user_emotion"],
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
        f"│       Express   : {'✓' if thought.get('should_express') else '✗'}",
        f"│  [S3] Reasoning : {thought.get('reasoning','–')}",
        f"│       Search    : {'✓ ' + thought.get('search_query','') if thought.get('need_search') else '✗'}",
        f"│       Recall    : {thought.get('recall_topic') or '–'} (source: {thought.get('recall_source','none')})",
        f"│       UseMemory : {'✓' if thought.get('use_memory') else '✗'}",
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
