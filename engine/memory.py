"""
engine/memory.py — Public API untuk sistem memori Asta.
"""

from pathlib import Path
from .memory_system import SemanticMemory, EpisodicMemory, CoreMemory, HybridMemory

MEMORY_DIR = Path("memory")
MEMORY_DIR.mkdir(exist_ok=True)

# Instance global
semantic_memory = SemanticMemory(MEMORY_DIR)
episodic_memory = EpisodicMemory(MEMORY_DIR)
core_memory = CoreMemory(MEMORY_DIR)
hybrid_memory = HybridMemory(episodic=episodic_memory, core=core_memory)

# Simpan referensi semantic ke hybrid agar web_tools bisa akses
hybrid_memory.semantic = semantic_memory


# ─── Semantic / Identity ──────────────────────────────────────────────────────

def remember_identity(key: str, value):
    semantic_memory.add_fact(key, value)

def get_identity(key: str):
    return semantic_memory.get_fact(key)

def get_all_identities() -> dict:
    return semantic_memory.get_all_facts()


# ─── Episodic ─────────────────────────────────────────────────────────────────

def add_episodic(conversation: list, llm_summary: str = ""):
    episodic_memory.add(conversation, llm_summary=llm_summary)

def search_episodic(query: str, top_k: int = 3) -> list:
    return episodic_memory.search(query, top_k=top_k)

def get_last_episodic_sessions(n: int = 3) -> list:
    return episodic_memory.get_last_n(n)


# ─── Core Memory ─────────────────────────────────────────────────────────────

def get_core_memory() -> str:
    return core_memory.get_summary()

def save_core_memory(text: str):
    core_memory.update_summary(text, async_save=False)


# ─── Hybrid ──────────────────────────────────────────────────────────────────

def get_hybrid_memory() -> HybridMemory:
    return hybrid_memory

def get_memory_context(query: str = "", max_chars: int = 1200) -> str:
    return hybrid_memory.get_context(current_query=query, max_chars=max_chars)
