_THOUGHT_SYSTEM_TEMPLATE = (
    "Analisis input. Pengguna={user_name}, AI=Asta.\n"
    "\"aku\" dari user = {user_name}.\n\n"
    "NEED_SEARCH=yes jika: info realtime, sebut 'cari/cek', karya spesifik yg model tidak yakin.\n"
    "RECALL_TOPIC: isi jika topik pernah dibahas. Pakai nama \"{user_name}\" jika tentang user. Kosong jika tidak ada.\n\n"
    "FORMAT (stop setelah NOTE):\n"
    "NEED_SEARCH: yes/no\n"
    "SEARCH_QUERY: <query atau kosong>\n"
    "RECALL_TOPIC: <topik atau kosong>\n"
    "USER_EMOTION: netral/sedih/cemas/marah/senang/romantis\n"
    "EMOTION_CONFIDENCE: rendah/sedang/tinggi\n"
    "TONE: romantic/casual/informative\n"
    "NOTE: <maks 6 kata>"
)

_STOP_TOKENS = ["\n\n", "---", "</thought>", "###", "Input "]


def run_thought_pass(
    llm,
    user_input: str,
    memory_context: str,
    recent_context: str = "",
    web_search_enabled: bool = True,
    max_tokens: int = 50,
    user_name: str = "Aditiya",
    emotion_state: str = "",
) -> dict:
    thought_system = _THOUGHT_SYSTEM_TEMPLATE.format(user_name=user_name)

    mem_hint = ""
    if memory_context:
        mem_hint = memory_context.strip().splitlines()[0][:60]

    ctx_block = f"Konteks:\n{recent_context}\n\n" if recent_context else ""
    emo_block = f"Emosi: {emotion_state}\n" if emotion_state else ""

    prompt = (
        f"{thought_system}\n\n"
        f"Memori: {mem_hint or '(kosong)'}\n"
        f"Search: {'ya' if web_search_enabled else 'tidak'}\n\n"
        f"{ctx_block}"
        f"{emo_block}"
        f"Input {user_name}: \"{user_input[:150]}\"\n\n"
        f"NEED_SEARCH:"
    )

    try:
        llm.reset()

        result = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.01,
            top_p=0.9,
            stop=_STOP_TOKENS,
            echo=False,
        )
        raw = "NEED_SEARCH:" + result["choices"][0]["text"].strip()
    except Exception as e:
        print(f"[Thought] Pass gagal: {e}")
        raw = ""
    finally:
        try:
            llm.reset()
        except Exception:
            pass

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


def extract_recent_context(conversation_history: list, n: int = 2) -> str:
    recent = conversation_history[-n:] if len(conversation_history) >= n else conversation_history
    lines = []
    for msg in recent:
        role = "Kamu" if msg["role"] == "user" else "Asta"
        content = msg["content"]
        if content:
            lines.append(f"{role}: {content[:100]}")
    return "\n".join(lines)


def format_thought_debug(thought: dict, web_result: str = "") -> str:
    lines = []
    lines.append("┌─ [Thought] ───────────────────────────────────────────")
    raw = thought.get("raw", "").strip()
    if raw:
        for line in raw.splitlines():
            lines.append(f"│  {line}")
    else:
        lines.append("│  (kosong / gagal)")

    lines.append("├─ [Parsed] ────────────────────────────────────────────")
    lines.append(f"│  Search  : {'✓' if thought['need_search'] else '✗'}")
    if thought["need_search"] and thought.get("search_query"):
        lines.append(f"│  Query   : {thought['search_query']}")
    lines.append(f"│  Recall  : {thought.get('recall_topic') or '–'}")
    lines.append(f"│  Emotion : {thought.get('user_emotion','netral')} ({thought.get('emotion_confidence','rendah')})")
    lines.append(f"│  Tone    : {thought.get('tone','–')}")
    lines.append(f"│  Note    : {thought.get('note') or '–'}")

    if thought["need_search"]:
        lines.append("├─ [Web Search] ────────────────────────────────────────")
        if web_result and not web_result.startswith("[INFO]"):
            for line in web_result[:400].splitlines():
                if line.strip():
                    lines.append(f"│  {line}")
            if len(web_result) > 400:
                lines.append(f"│  ... ({len(web_result)} chars total)")
        elif web_result.startswith("[INFO]"):
            lines.append("│  ✗ Fetch gagal")
        else:
            lines.append("│  ✗ Tidak ada hasil")

    lines.append("└───────────────────────────────────────────────────────")
    return "\n".join(lines)
