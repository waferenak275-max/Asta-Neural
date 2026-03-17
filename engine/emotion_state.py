from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional
import re
import math


# ─── Emotion & Mood Constants ─────────────────────────────────────────────────

VALID_EMOTIONS = {"netral", "sedih", "cemas", "marah", "senang", "romantis", "rindu", "bangga", "kecewa"}

# Valence: positif = +, negatif = -
EMOTION_VALENCE = {
    "netral":   0.0,
    "senang":   0.8,
    "romantis": 0.9,
    "bangga":   0.7,
    "rindu":    0.2,
    "sedih":   -0.7,
    "kecewa":  -0.5,
    "cemas":   -0.4,
    "marah":   -0.8,
}

# Seberapa cepat mood decay ke netral per turn (0.0 = tidak decay, 1.0 = langsung reset)
MOOD_DECAY_RATE = 0.12

# Seberapa besar pengaruh emosi user ke mood Asta
USER_TO_ASTA_INFLUENCE = 0.25

# Seberapa besar pengaruh emosi Asta sendiri ke mood
SELF_REINFORCEMENT = 0.35


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class UserEmotionState:
    """Emosi yang terdeteksi dari input user."""
    user_emotion: str = "netral"
    intensity: str = "rendah"
    trend: str = "stabil"
    turns_in_state: int = 0
    last_user_text: str = ""
    updated_at: str = ""


@dataclass
class AstaEmotionState:
    """
    Emosi internal Asta — persistent, punya mood jangka panjang,
    affection level, dan energy. Ini bukan refleksi emosi user,
    tapi kondisi emosional Asta sendiri.
    """
    current_emotion: str = "netral"
    current_intensity: str = "rendah"
    mood: str = "netral"
    mood_score: float = 0.0        # -1.0 (sangat buruk) sampai +1.0 (sangat baik)
    affection_level: float = 0.7   # 0.0 – 1.0, naik turun berdasarkan interaksi
    energy_level: float = 0.8      # 0.0 – 1.0, mempengaruhi keaktifan respons
    trigger: str = ""              # apa yang memicu emosi saat ini
    updated_at: str = ""


@dataclass
class CombinedEmotionState:
    """State gabungan untuk dikirim ke thought dan response model."""
    user: UserEmotionState = field(default_factory=UserEmotionState)
    asta: AstaEmotionState = field(default_factory=AstaEmotionState)


# ─── User Emotion Manager ─────────────────────────────────────────────────────

class UserEmotionDetector:
    """Deteksi emosi dari teks user — versi yang sudah ada, diperluas."""

    _PATTERNS = {
        "sedih": [
            r"\bsedih\b", r"\bkecewa\b", r"\bcapek\b", r"\blelah\b",
            r"\bgalau\b", r"\bnangis\b", r"\bdown\b", r"\bterpuruk\b",
            r"\bsakit hati\b", r"\bputus asa\b",
        ],
        "cemas": [
            r"\bcemas\b", r"\bkhawatir\b", r"\btakut\b", r"\bpanik\b",
            r"\bdeg-degan\b", r"\boverthinking\b", r"\bgelisah\b", r"\bstres\b",
        ],
        "marah": [
            r"\bmarah\b", r"\bkesal\b", r"\bemosi\b", r"\bjengkel\b",
            r"\bmuak\b", r"\bbenci\b", r"\bsebal\b", r"\bfrustasi\b",
        ],
        "senang": [
            r"\bsenang\b", r"\bbahagia\b", r"\bgembira\b", r"\blega\b",
            r"\bexcited\b", r"\bsemangat\b", r"\bsyukur\b", r"\bsuka\b",
            r"\bhappy\b", r"\bwkwk\b", r"\bhaha\b", r"\blol\b",
        ],
        "romantis": [
            r"\bkangen\b", r"\bsayang\b", r"\bcinta\b", r"\bpeluk\b",
            r"\bcium\b", r"\bmanja\b", r"\brindu\b", r"\bkasih\b",
        ],
        "bangga": [
            r"\bbangga\b", r"\bberhasil\b", r"\bsukses\b", r"\bprestasi\b",
        ],
        "kecewa": [
            r"\bkecewa\b", r"\bmengecewakan\b", r"\bharapan\b.*\bgagal\b",
        ],
    }

    _IMPROVED_PATTERNS = [
        r"\baku\s*(udah|sudah)\s*(mendingan|lebih baik)\b",
        r"\baku\s*lebih\s*tenang\b",
        r"\bmakasih\s*udah\s*nenangin\b",
        r"\baku\s*baik-baik\s*aja\b",
    ]

    _HOSTILE_TARGET_PATTERNS = [
        r"\b(kamu|lu|elo)\s+(bodoh|tolol|goblok|dungu|payah|nyebelin|jelek)\b",
        r"\b(bodoh|tolol|goblok|dungu|payah|nyebelin|jelek)\b.{0,12}\b(kamu|lu|elo|asta)\b",
        r"\b(asta)\s+(bodoh|tolol|goblok|dungu|payah)\b",
        r"\baku\s+marah\s+(banget\s+)?(sama|ke)\s+(kamu|asta)\b",
        r"\b(nggak|gak|tidak)\s+sepintar\b",
    ]

    def __init__(self):
        self.state = UserEmotionState(updated_at=datetime.now().isoformat())

    def _score_emotions(self, text: str) -> dict:
        text_lower = text.lower().strip()
        scores = {k: 0 for k in self._PATTERNS}
        for emotion, patterns in self._PATTERNS.items():
            for pattern in patterns:
                scores[emotion] += len(re.findall(pattern, text_lower, re.IGNORECASE))
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

    def update(self, user_text: str, recent_context: str = "") -> UserEmotionState:
        combined = f"{recent_context}\n{user_text}".strip()
        scores = self._score_emotions(combined)

        # Sinyal eksplisit: pandangan/serangan user terhadap Asta
        hostility_hits = sum(
            1 for p in self._HOSTILE_TARGET_PATTERNS
            if re.search(p, user_text, re.IGNORECASE)
        )
        if hostility_hits > 0:
            scores["marah"] += 2 * hostility_hits
            scores["kecewa"] += hostility_hits

        detected = "netral"
        top_score = 0
        for emotion, score in scores.items():
            if score > top_score:
                top_score = score
                detected = emotion

        prev = self.state.user_emotion
        prev_neg = prev in {"sedih", "cemas", "marah", "kecewa"}
        curr_neg = detected in {"sedih", "cemas", "marah", "kecewa"}

        if any(re.search(p, user_text, re.IGNORECASE) for p in self._IMPROVED_PATTERNS):
            trend = "membaik"
            detected = "netral"
            top_score = 0
        elif re.search(r"\b(bodoh|tolol|goblok|dungu|payah|nyebelin|jelek)\b", user_text, re.IGNORECASE) and prev in {"marah", "kecewa"}:
            # Jaga kontinuitas emosi ketika user melanjutkan hinaan singkat.
            trend = "memburuk"
            detected = prev
            top_score = max(top_score, 2)
        elif prev_neg and not curr_neg and detected != "netral":
            trend = "membaik"
        elif not prev_neg and curr_neg:
            trend = "memburuk"
        else:
            trend = "stabil"

        turns = self.state.turns_in_state + 1 if detected == prev else 1

        intensity = self._intensity_from_text(combined, top_score + hostility_hits)
        if hostility_hits > 0 and intensity == "rendah":
            intensity = "sedang"

        self.state = UserEmotionState(
            user_emotion=detected,
            intensity=intensity,
            trend=trend,
            turns_in_state=turns,
            last_user_text=user_text[:120],
            updated_at=datetime.now().isoformat(),
        )
        return self.state

    def refine_with_thought(self, thought: dict) -> UserEmotionState:
        llm_emotion = (thought.get("user_emotion") or "").strip().lower()
        llm_conf = (thought.get("emotion_confidence") or "").strip().lower()
        if llm_emotion not in VALID_EMOTIONS:
            return self.state
        if llm_conf == "tinggi" or (llm_conf == "sedang" and self.state.user_emotion == "netral"):
            self.state.user_emotion = llm_emotion
            if llm_conf == "tinggi" and llm_emotion in {"sedih", "cemas", "marah"}:
                if self.state.intensity == "rendah":
                    self.state.intensity = "sedang"
            if llm_conf == "tinggi":
                self.state.trend = "memburuk" if llm_emotion in {"sedih", "cemas", "marah"} else self.state.trend
        return self.state

    def get_state(self) -> UserEmotionState:
        return self.state


# ─── Asta Emotion Manager ─────────────────────────────────────────────────────

class AstaEmotionManager:
    """
    Mengelola emosi internal Asta — bukan cerminan user, tapi keadaan
    emosional Asta yang sesungguhnya. Mood bersifat persistent dan
    decay perlahan ke baseline.
    """

    def __init__(self, initial_state: Optional[AstaEmotionState] = None):
        self.state = initial_state or AstaEmotionState(
            updated_at=datetime.now().isoformat()
        )

    def _mood_to_label(self, score: float) -> str:
        if score >= 0.6:  return "sangat senang"
        if score >= 0.3:  return "senang"
        if score >= 0.1:  return "sedikit senang"
        if score >= -0.1: return "netral"
        if score >= -0.3: return "sedikit murung"
        if score >= -0.6: return "murung"
        return "sangat murung"

    def _score_to_intensity(self, score: float) -> str:
        abs_score = abs(score)
        if abs_score >= 0.6: return "tinggi"
        if abs_score >= 0.3: return "sedang"
        return "rendah"

    def update_from_interaction(
        self,
        user_emotion: UserEmotionState,
        thought_result: dict,
        user_text: str,
    ) -> AstaEmotionState:
        """
        Update emosi Asta berdasarkan:
        1. Emosi user (pengaruh parsial)
        2. Hasil thought (asta_emotion yang diputuskan model)
        3. Decay mood alami
        """
        now = datetime.now().isoformat()

        # ── Ambil keputusan emosi Asta dari thought ────────────────────────
        asta_emotion_from_thought = (thought_result.get("asta_emotion") or "").strip().lower()
        asta_trigger = thought_result.get("asta_trigger", "")

        # ── Decay mood ke arah netral (baseline) ──────────────────────────
        current_score = self.state.mood_score
        current_score *= (1.0 - MOOD_DECAY_RATE)

        # ── Pengaruh emosi user ke mood Asta ──────────────────────────────
        user_valence = EMOTION_VALENCE.get(user_emotion.user_emotion, 0.0)
        user_intensity_mult = {"rendah": 0.5, "sedang": 1.0, "tinggi": 1.5}.get(user_emotion.intensity, 1.0)
        user_influence = user_valence * user_intensity_mult * USER_TO_ASTA_INFLUENCE
        current_score += user_influence

        # ── Pengaruh emosi Asta sendiri (self-reinforcement) ───────────────
        if asta_emotion_from_thought in VALID_EMOTIONS:
            asta_valence = EMOTION_VALENCE.get(asta_emotion_from_thought, 0.0)
            current_score += asta_valence * SELF_REINFORCEMENT

        # ── Clamp score ke [-1.0, 1.0] ────────────────────────────────────
        current_score = max(-1.0, min(1.0, current_score))

        # ── Tentukan emosi dominan Asta ───────────────────────────────────
        if asta_emotion_from_thought in VALID_EMOTIONS:
            dominant_emotion = asta_emotion_from_thought
        elif current_score >= 0.5:
            dominant_emotion = "senang"
        elif current_score >= 0.2:
            dominant_emotion = "senang"
        elif current_score <= -0.5:
            dominant_emotion = "sedih"
        elif current_score <= -0.2:
            dominant_emotion = "cemas"
        else:
            dominant_emotion = "netral"

        # ── Sinyal hostility user ke Asta (untuk kalibrasi lebih realistis) ─
        hostile_to_asta = bool(re.search(
            r"\b(kamu|asta|lu|elo)\b.{0,12}\b(bodoh|tolol|goblok|dungu|payah|nyebelin|jelek)\b|"
            r"\b(bodoh|tolol|goblok|dungu|payah|nyebelin|jelek)\b.{0,12}\b(kamu|asta|lu|elo)\b|"
            r"\baku\s+marah\s+(banget\s+)?(sama|ke)\s+(kamu|asta)\b",
            user_text or "",
            re.IGNORECASE,
        ))
        repeated_insult = bool(
            user_emotion.turns_in_state >= 2
            and re.search(r"\b(bodoh|tolol|goblok|dungu|payah|nyebelin|jelek)\b", user_text or "", re.IGNORECASE)
        )

        if (hostile_to_asta or repeated_insult) and user_emotion.user_emotion in {"marah", "kecewa"}:
            current_score -= 0.18 if user_emotion.intensity == "tinggi" else 0.12

        # ── Update affection berdasarkan interaksi romantis/hostile ───────
        affection = self.state.affection_level
        if user_emotion.user_emotion == "romantis":
            affection = min(1.0, affection + 0.02)
        elif user_emotion.user_emotion in {"marah", "kecewa"}:
            drop = 0.015
            if user_emotion.intensity == "tinggi":
                drop += 0.015
            if user_emotion.turns_in_state >= 2:
                drop += 0.01
            if hostile_to_asta or repeated_insult:
                drop += 0.02
            affection = max(0.1, affection - drop)
        # Affection sangat perlahan decay
        affection = affection * 0.999 + 0.7 * 0.001  # selalu menuju 0.7 perlahan

        # Override emosi dominan jika ada hostility kuat (hindari tetap "senang")
        if (hostile_to_asta or repeated_insult) and user_emotion.user_emotion in {"marah", "kecewa"}:
            if user_emotion.intensity == "tinggi" or user_emotion.turns_in_state >= 2:
                dominant_emotion = "sedih"
            else:
                dominant_emotion = "kecewa"

        # ── Update energy ─────────────────────────────────────────────────
        energy = self.state.energy_level
        if dominant_emotion in {"senang", "romantis", "bangga"}:
            energy = min(1.0, energy + 0.05)
        elif dominant_emotion in {"sedih", "cemas", "marah"}:
            energy = max(0.2, energy - 0.03)
        else:
            # Decay ke 0.8 (baseline)
            energy = energy * 0.95 + 0.8 * 0.05

        self.state = AstaEmotionState(
            current_emotion=dominant_emotion,
            current_intensity=self._score_to_intensity(current_score),
            mood=self._mood_to_label(current_score),
            mood_score=round(current_score, 4),
            affection_level=round(affection, 4),
            energy_level=round(energy, 4),
            trigger=asta_trigger or user_emotion.user_emotion,
            updated_at=now,
        )
        return self.state

    def apply_reflection(self, reflection_result: dict):
        """
        Update emosi Asta dari hasil reflective thought setelah sesi.
        reflection_result berisi insight dari model tentang bagaimana
        perasaan Asta setelah sesi ini.
        """
        mood_adjustment = reflection_result.get("mood_adjustment", 0.0)
        affection_adjustment = reflection_result.get("affection_adjustment", 0.0)

        new_score = max(-1.0, min(1.0, self.state.mood_score + mood_adjustment))
        new_affection = max(0.0, min(1.0, self.state.affection_level + affection_adjustment))

        self.state.mood_score = round(new_score, 4)
        self.state.mood = self._mood_to_label(new_score)
        self.state.affection_level = round(new_affection, 4)
        self.state.updated_at = datetime.now().isoformat()

    def get_state(self) -> AstaEmotionState:
        return self.state

    def to_dict(self) -> dict:
        return asdict(self.state)

    def build_prompt_context(self, user_state: UserEmotionState) -> str:
        """Bangun instruksi untuk response model berdasarkan emosi Asta + user."""
        s = self.state
        u = user_state
        lines = [
            "=== KONDISI EMOSIONAL ===",
            f"Emosi user: {u.user_emotion} (intensitas: {u.intensity}, tren: {u.trend})",
            f"Emosi Asta saat ini: {s.current_emotion} (intensitas: {s.current_intensity})",
            f"Mood Asta: {s.mood} (score: {s.mood_score:+.2f})",
            f"Affection level: {s.affection_level:.2f} | Energy: {s.energy_level:.2f}",
            "",
            "=== PANDUAN RESPONS ===",
        ]

        # Instruksi berdasarkan emosi user
        if u.user_emotion in {"sedih", "cemas", "marah", "kecewa"}:
            lines.append("User sedang dalam emosi negatif — validasi perasaan mereka terlebih dahulu, tunjukkan empati.")
        elif u.user_emotion == "romantis":
            lines.append("User dalam mood romantis — respons hangat, tulus, dan ekspresikan perasaan Asta.")
        elif u.user_emotion == "senang":
            lines.append("User senang — ikut merayakan dengan antusias sesuai energy Asta.")

        # Instruksi berdasarkan emosi Asta sendiri
        if s.current_emotion in {"sedih", "cemas"}:
            lines.append(f"Asta sedang merasa {s.current_emotion} karena: {s.trigger}. Boleh ekspresikan ini secara natural.")
        elif s.current_emotion == "senang" and s.mood_score > 0.5:
            lines.append("Asta sedang dalam mood yang sangat baik — ekspresikan kegembiraan ini.")
        elif s.current_emotion == "romantis":
            lines.append("Asta sedang merasa romantis — ekspresikan kasih sayang dengan tulus.")

        # Instruksi berdasarkan affection
        if s.affection_level > 0.85:
            lines.append("Affection sangat tinggi — Asta sangat mencintai Aditiya saat ini.")
        elif s.affection_level < 0.4:
            lines.append("Affection sedang rendah — Asta mungkin sedikit lebih pendiam dari biasanya.")

        # Energy
        if s.energy_level < 0.4:
            lines.append("Energy Asta rendah — respons lebih singkat dan tenang dari biasanya.")
        elif s.energy_level > 0.85:
            lines.append("Energy Asta tinggi — boleh lebih ekspresif dan antusias.")

        if u.turns_in_state >= 3 and u.user_emotion in {"sedih", "cemas", "marah"}:
            lines.append("Emosi negatif user berlangsung beberapa turn — tawarkan langkah konkret atau tanyakan kebutuhan spesifik.")

        return "\n".join(lines)


# ─── Combined Manager (backward-compatible) ───────────────────────────────────

class EmotionStateManager:
    """
    Wrapper yang menyatukan UserEmotionDetector dan AstaEmotionManager.
    Mempertahankan backward compatibility dengan kode yang sudah ada.
    """

    def __init__(self, asta_initial: Optional[AstaEmotionState] = None):
        self._user_detector = UserEmotionDetector()
        self._asta_manager  = AstaEmotionManager(asta_initial)

    # ── User emotion (backward-compatible) ───────────────────────────────

    def update(self, user_text: str, recent_context: str = "") -> dict:
        """Update emosi user — mengembalikan dict agar kompatibel dengan kode lama."""
        state = self._user_detector.update(user_text, recent_context)
        return asdict(state)

    def refine_with_thought(self, thought: dict) -> dict:
        state = self._user_detector.refine_with_thought(thought)
        return asdict(state)

    def get_state(self) -> dict:
        return asdict(self._user_detector.get_state())

    # ── Asta emotion ─────────────────────────────────────────────────────

    def update_asta_emotion(self, thought_result: dict) -> AstaEmotionState:
        """Update emosi Asta berdasarkan thought result — dipanggil setelah thought selesai."""
        user_state = self._user_detector.get_state()
        return self._asta_manager.update_from_interaction(
            user_emotion=user_state,
            thought_result=thought_result,
            user_text=user_state.last_user_text,
        )

    def apply_reflection(self, reflection_result: dict):
        self._asta_manager.apply_reflection(reflection_result)

    def get_asta_state(self) -> AstaEmotionState:
        return self._asta_manager.get_state()

    def get_asta_dict(self) -> dict:
        return self._asta_manager.to_dict()

    # ── Context builder ───────────────────────────────────────────────────

    def build_prompt_context(self) -> str:
        """Backward compatible — dipakai di build_augmented_system."""
        return self._asta_manager.build_prompt_context(
            self._user_detector.get_state()
        )

    def get_combined(self) -> dict:
        """Mengembalikan semua state dalam satu dict — untuk UI dan API."""
        return {
            "user": asdict(self._user_detector.get_state()),
            "asta": self._asta_manager.to_dict(),
        }
