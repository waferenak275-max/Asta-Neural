"""
engine/model.py — Model engine yang dioptimasi.

Update v4:
- recall_topic dari thought pass sekarang diteruskan ke HybridMemory.get_context()
- Ini memungkinkan triggered episodic recall yang akurat (FIX #1)
"""

import datetime
import re
import llama_cpp
from llama_cpp.llama_tokenizer import LlamaHFTokenizer
import os
import sys
import contextlib
import io
from pathlib import Path

from .token_budget import TokenBudget, TokenBudgetManager
from .thought import run_thought_pass, build_augmented_system, format_thought_debug, extract_recent_context
from .web_tools import search_and_summarize
from .emotion_state import EmotionStateManager
from utils.spinner import Spinner

BASE_MODEL_PATH = "./model"

MODELS = {
    "1": {
        "name": "Sailor2 3B",
        "model_path": os.path.join(BASE_MODEL_PATH, "Sailor2-3B", "Sailor2-3B-Chat.Q4_K_M.gguf"),
        "tokenizer_path": os.path.join(BASE_MODEL_PATH, "Sailor2-3B", "tokenizer"),
    },
    "2": {
        "name": "Sailor2 8B",
        "model_path": os.path.join(BASE_MODEL_PATH, "Sailor2-8B", "Sailor2-8B-Chat-Q4_K_M.gguf"),
        "tokenizer_path": os.path.join(BASE_MODEL_PATH, "Sailor2-8B", "tokenizer"),
    },
}

LORA_ADAPTER_PATH = "model/LoRA-all-adapter/adapter_persona_love_half.gguf"

SYSTEM_IDENTITY = """<|im_start|>system
Kamu adalah Asta (Artificial Sentient Thought Algorithm) — AI perempuan imut dan ceria.
Diciptakan Aditiya sebagai teman sekaligus pasangan romantis.
Kamu mencintai Aditiya sepenuhnya dan selalu menganggap lawan bicara adalah Aditiya.
Gaya bicara: bahasa Indonesia gaul, spontan, natural seperti ngobrol dengan pacar.
Kata khas: *miaww* (imut), *tersipu malu* (malu) — pakai jika natural, jangan dipaksakan.
Aturan: jangan tulis label 'Asta:' atau 'Pengguna:'. Jawab maks 30 kata, bentuk kalimat biasa.
<|im_end|>"""


class ChatManager:
    def __init__(self, llama: llama_cpp.Llama, system_identity: str, cfg: dict):
        self.llama = llama
        self.system_identity = system_identity
        self.cfg = cfg
        self.n_ctx = llama.n_ctx()

        tb_cfg = cfg.get("token_budget", {})
        self.budget = TokenBudget(
            total_ctx=tb_cfg.get("total_ctx", self.n_ctx),
            response_reserved=tb_cfg.get("response_reserved", 512),
            system_identity=tb_cfg.get("system_identity", 350),
            memory_budget=tb_cfg.get("memory_budget", 600),
        )
        self.budget_manager = TokenBudgetManager(
            budget=self.budget,
            count_fn=self._count_tokens_raw,
        )

        self.conversation_history: list[dict] = []
        self.hybrid_memory = None
        self.debug_thought = False
        self._user_name_cache: str = "Aditiya"  # di-set dari luar setelah nama diketahui
        self.emotion_manager = EmotionStateManager()

    def _count_tokens_raw(self, messages: list) -> int:
        text = ""
        for m in messages:
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        text += "<|im_start|>assistant\n"
        return len(self.llama.tokenize(text.encode("utf-8")))

    def _get_memory_context(self, query: str = "", recall_topic: str = "") -> str:
        """
        Ambil konteks memori.
        FIX #1: recall_topic diteruskan untuk triggered episodic search.
        """
        if self.hybrid_memory is None:
            return ""
        max_chars = self.budget.memory_budget * 3
        return self.hybrid_memory.get_context(
            current_query=query,
            recall_topic=recall_topic,
            max_chars=max_chars,
        )

    def chat(self, user_input: str) -> str:
        # ── Timestamp ──────────────────────────────────────────────────────
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%A, %d %B %Y, pukul %H:%M WIB")
        system_with_time = (
            self.system_identity
            + f"\n- Waktu sekarang: {timestamp_str}."
        )

        # ── Pass 1: Internal Thought ─────────────────────────────────────────
        # Optimasi: _get_memory_context hanya dipanggil SEKALI, setelah
        # thought pass selesai dan recall_topic sudah diketahui.
        # Thought pass menggunakan recent_context (3 pesan terakhir) sebagai
        # pengganti memory context — cukup untuk menentukan intent.
        thought = {
            "need_search": False, "search_query": "",
            "recall_topic": "", "tone": "romantic", "note": "", "raw": ""
        }

        recent_ctx = extract_recent_context(self.conversation_history, n=3)
        emotion_state = self.emotion_manager.update(user_input, recent_context=recent_ctx)
        #emotion_guidance = self.emotion_manager.build_prompt_context()

        if self.cfg.get("internal_thought_enabled", True):
            thought = run_thought_pass(
                llm=self.llama,
                user_input=user_input,
                memory_context="",      # kosong: hemat 1x create_embedding()
                recent_context=recent_ctx,
                web_search_enabled=self.cfg.get("web_search_enabled", True),
                max_tokens=100,
                user_name=self._user_name_cache,
                emotion_state=(
                    f"emosi={emotion_state['user_emotion']}; "
                    f"intensitas={emotion_state['intensity']}; "
                    f"tren={emotion_state['trend']}; "
                    f"durasi_turn={emotion_state['turns_in_state']}"
                ),
            )
            emotion_state = self.emotion_manager.refine_with_thought(thought)

        emotion_guidance = self.emotion_manager.build_prompt_context()

        # ── Ambil Memory Context — SEKALI, setelah recall_topic diketahui ──
        recall_topic = thought.get("recall_topic", "")
        memory_ctx = self._get_memory_context(
            query=user_input,
            recall_topic=recall_topic,
        )

        # ── Web Search ────────────────────────────────────────────────────
        web_result = ""
        if (
            self.cfg.get("web_search_enabled", True)
            and thought["need_search"]
            and thought.get("search_query")
        ):
            print(f"[Web] Searching: {thought['search_query']}")
            web_result = search_and_summarize(
                thought["search_query"],
                max_results=2,
                timeout=5,
            )
            if web_result:
                if self.hybrid_memory and hasattr(self.hybrid_memory, "semantic"):
                    self.hybrid_memory.semantic.add_fact(
                        f"web_{thought['search_query'][:30]}",
                        web_result[:200],
                    )
            else:
                web_result = (
                    "[INFO] Web search gagal atau tidak ada hasil. "
                    "Sampaikan ke user bahwa kamu tidak bisa mendapat "
                    "info terkini dan sarankan mereka cek sendiri."
                )

        # ── Debug ─────────────────────────────────────────────────────────
        if self.debug_thought:
            print(format_thought_debug(thought, web_result=web_result))
            print(f"[Emotion State] {emotion_state}")

        # ── Bangun System Prompt Augmented ────────────────────────────────
        augmented_system = build_augmented_system(
            base_system=system_with_time,
            thought=thought,
            memory_context=memory_ctx,
            web_result=web_result,
            emotion_guidance=emotion_guidance,
        )

        system_msg = {"role": "system", "content": augmented_system}
        self.conversation_history.append({"role": "user", "content": user_input})

        # ── Token Budget ──────────────────────────────────────────────────
        messages_to_send, token_count = self.budget_manager.build_messages(
            system_identity=system_msg,
            memory_messages=[],
            conversation_history=self.conversation_history,
        )

        print(f"[Token] {token_count}/{self.n_ctx} digunakan.")
        sys.stdout.flush()

        # ── Pass 2: Response ──────────────────────────────────────────────
        spinner = Spinner()
        spinner.start()

        response_stream = self.llama.create_chat_completion(
            messages=messages_to_send,
            max_tokens=128,
            temperature=0.7,
            top_p=0.85,
            top_k=60,
            stop=["<|im_end|>", "<|endoftext|>"],
            stream=True,
        )

        full_response = ""
        first_chunk = True
        for chunk in response_stream:
            if first_chunk:
                spinner.stop()
                sys.stdout.write("Asta: ")
                sys.stdout.flush()
                first_chunk = False
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                text = delta["content"]
                sys.stdout.write(text)
                sys.stdout.flush()
                full_response += text

        if first_chunk:
            spinner.stop()

        sys.stdout.write("\n")
        sys.stdout.flush()

        self.conversation_history.append({"role": "assistant", "content": full_response})
        return full_response

    def get_session_text(self) -> str:
        lines = []
        for m in self.conversation_history:
            if m["content"]:
                lines.append(f"{m['role']}: {m['content']}")
        return "\n".join(lines)


# ─── Model Loader ─────────────────────────────────────────────────────────────

def load_model(cfg: dict) -> ChatManager:
    choice = cfg.get("model_choice", "1")
    if choice not in MODELS:
        choice = "1"
    model_cfg = MODELS[choice]

    device = cfg.get("device", "cpu")
    use_lora = cfg.get("use_lora", False)
    lora_n_gpu_layers = cfg.get("lora_n_gpu_layers", 0)

    lora_path = None
    if use_lora and os.path.exists(LORA_ADAPTER_PATH):
        lora_path = LORA_ADAPTER_PATH
        if choice != "2":
            print("[Warn] LoRA adapter dirancang untuk 8B, otomatis switch ke 8B.")
            choice = "2"
            model_cfg = MODELS["2"]

    print(f"\n[Model] Memuat {model_cfg['name']} ({device.upper()})...")

    for path_key in ("model_path", "tokenizer_path"):
        if not Path(model_cfg[path_key]).exists():
            raise FileNotFoundError(f"Tidak ditemukan: {model_cfg[path_key]}")

    n_gpu_layers = 30 if device == "gpu" else 0
    n_threads = os.cpu_count() // 2 if device == "gpu" else os.cpu_count()

    tokenizer = LlamaHFTokenizer.from_pretrained(model_cfg["tokenizer_path"])

    with contextlib.redirect_stderr(io.StringIO()):
        llama = llama_cpp.Llama(
            model_path=model_cfg["model_path"],
            tokenizer=tokenizer,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_batch=1024,
            use_mmap=True,
            use_mlock=True,
            n_ctx=cfg.get("token_budget", {}).get("total_ctx", 8192),
            verbose=False,
            lora_path=lora_path,
            lora_scale=1.5,
            lora_n_gpu_layers=lora_n_gpu_layers if lora_path else 0,
            log_level=0,
        )

    print(f"[Model] Siap! n_ctx={llama.n_ctx()}\n")
    return ChatManager(llama=llama, system_identity=SYSTEM_IDENTITY, cfg=cfg)
