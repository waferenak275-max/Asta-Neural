from dataclasses import dataclass, asdict
from datetime import datetime
import re

@dataclass
class EmotionState:
    user_emotion: str = "netral"
    intensity: str = "rendah"
    trend: str = "stabil"
    turns_in_state: int = 0
    last_user_text: str = ""
    updated_at: str = ""


class EmotionStateManager:
    _PATTERNS = {
        "sedih": [
            r"\bsedih\b", r"\bkecewa\b", r"\bcapek\b", r"\blelah\b",
            r"\bgalau\b", r"\bnangis\b", r"\bdown\b", r"\bterpuruk\b",
        ],
        "cemas": [
            r"\bcemas\b", r"\bkhawatir\b", r"\btakut\b", r"\bpanik\b",
            r"\bdeg-degan\b", r"\boverthinking\b",
        ],
        "marah": [
            r"\bmarah\b", r"\bkesal\b", r"\bemosi\b", r"\bjengkel\b",
            r"\bmuak\b", r"\bbenci\b",
        ],
        "senang": [
            r"\bsenang\b", r"\bbahagia\b", r"\bgembira\b", r"\blega\b",
            r"\bexcited\b", r"\bsemangat\b", r"\bsyukur\b",
        ],
        "romantis": [
            r"\bkangen\b", r"\bsayang\b", r"\bcinta\b", r"\bpeluk\b",
            r"\bcium\b", r"\bmanja\b",
        ],
    }

    _IMPROVED_PATTERNS = [
        r"\baku\s*(udah|sudah)\s*(mendingan|lebih baik)\b",
        r"\baku\s*lebih\s*tenang\b",
        r"\bmakasih\s*udah\s*nenangin\b",
        r"\baku\s*baik-baik\s*aja\b",
    ]

    def __init__(self):
        self.state = EmotionState(updated_at=datetime.now().isoformat())

    def _score_emotions(self, text: str) -> dict:
        text = text.lower().strip()
        scores = {k: 0 for k in self._PATTERNS}

        for emotion, patterns in self._PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, flags=re.IGNORECASE)
                scores[emotion] += len(matches)

        if text.endswith("!"):
            scores["senang"] += 1
        if "..." in text:
            scores["sedih"] += 1
            scores["cemas"] += 1

        return scores

    def _intensity_from_text(self, text: str, score: int) -> str:
        if score >= 3 or re.search(r"\bbanget\b|\bbgt\b|\bparah\b|\bbener-bener\b", text, re.IGNORECASE):
            return "tinggi"
        if score >= 1:
            return "sedang"
        return "rendah"

    def update(self, user_text: str, recent_context: str = "") -> dict:
        combined_text = f"{recent_context}\n{user_text}".strip()
        scores = self._score_emotions(combined_text)

        detected_emotion = "netral"
        top_score = 0
        for emotion, score in scores.items():
            if score > top_score:
                top_score = score
                detected_emotion = emotion

        previous_emotion = self.state.user_emotion
        previous_negative = previous_emotion in {"sedih", "cemas", "marah"}
        current_negative = detected_emotion in {"sedih", "cemas", "marah"}

        if any(re.search(p, user_text, re.IGNORECASE) for p in self._IMPROVED_PATTERNS):
            trend = "membaik"
            detected_emotion = "netral"
            top_score = 0
        elif previous_negative and not current_negative and detected_emotion != "netral":
            trend = "membaik"
        elif not previous_negative and current_negative:
            trend = "memburuk"
        else:
            trend = "stabil"

        turns = self.state.turns_in_state + 1 if detected_emotion == previous_emotion else 1

        self.state = EmotionState(
            user_emotion=detected_emotion,
            intensity=self._intensity_from_text(combined_text, top_score),
            trend=trend,
            turns_in_state=turns,
            last_user_text=user_text[:120],
            updated_at=datetime.now().isoformat(),
        )
        return asdict(self.state)

    def get_state(self) -> dict:
        return asdict(self.state)


    def refine_with_thought(self, thought: dict) -> dict:
        llm_emotion = (thought.get("user_emotion") or "").strip().lower()
        llm_conf = (thought.get("emotion_confidence") or "").strip().lower()
        if llm_emotion not in {"netral", "sedih", "cemas", "marah", "senang", "romantis"}:
            return asdict(self.state)

        if llm_conf == "tinggi" or (llm_conf == "sedang" and self.state.user_emotion == "netral"):
            self.state.user_emotion = llm_emotion

            # Jika model yakin tinggi dan emosi negatif, minimalkan false-negative
            if llm_conf == "tinggi" and llm_emotion in {"sedih", "cemas", "marah"}:
                if self.state.intensity == "rendah":
                    self.state.intensity = "sedang"

            if llm_conf == "tinggi":
                self.state.trend = "memburuk" if llm_emotion in {"sedih", "cemas", "marah"} else self.state.trend

        return asdict(self.state)

    def build_prompt_context(self) -> str:
        s = self.state
        instructions = [
            "Gunakan state emosi pengguna sebagai sinyal gaya respons.",
            f"Emosi pengguna: {s.user_emotion} (intensitas: {s.intensity}, tren: {s.trend}).",
        ]

        if s.user_emotion in {"sedih", "cemas", "marah"}:
            instructions.append(
                "Prioritas: validasi perasaan pengguna, tenangkan dengan empati,"
                " lalu cek apakah pengguna mulai merasa lebih baik."
            )
        elif s.user_emotion == "senang":
            instructions.append("Prioritas: ikut merayakan dengan hangat dan tetap natural.")
        elif s.user_emotion == "romantis":
            instructions.append("Prioritas: respons romantis dan suportif, tetap singkat dan tulus.")
        else:
            instructions.append("Prioritas: respons seimbang, hangat, dan relevan dengan konteks.")

        if s.turns_in_state >= 3 and s.user_emotion in {"sedih", "cemas", "marah"}:
            instructions.append(
                "Emosi negatif berlangsung beberapa turn; tanyakan kebutuhan spesifik"
                " dan tawarkan langkah kecil yang actionable."
            )

        return "\n".join(instructions)
