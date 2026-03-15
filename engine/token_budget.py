"""
engine/token_budget.py — Mengelola alokasi token secara ketat.

Masalah sebelumnya: sliding window yang ada bisa membalik urutan pesan
dan tidak memperhitungkan memory injection secara terpisah.

Solusi: Token Budget yang eksplisit dengan slot terpisah untuk:
  - System identity (tetap, kecil)
  - Memory context (dinamis, dibatasi)
  - Conversation history (sisa budget)
  - Response reservation
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class TokenBudget:
    total_ctx: int = 8192
    response_reserved: int = 512
    system_identity: int = 350
    memory_budget: int = 600

    @property
    def conversation_budget(self) -> int:
        return (
            self.total_ctx
            - self.response_reserved
            - self.system_identity
            - self.memory_budget
        )

    @property
    def available_total(self) -> int:
        return self.total_ctx - self.response_reserved


class TokenBudgetManager:
    """
    Mengelola pemotongan history percakapan secara benar:
    - System prompt identity SELALU masuk (slot tetap)
    - Memory injection dibatasi memory_budget
    - History dipotong dari yang paling lama (bukan dibalik)
    """

    def __init__(self, budget: TokenBudget, count_fn):
        """
        count_fn: fungsi yang menerima list[dict] dan mengembalikan int token count
        """
        self.budget = budget
        self.count_fn = count_fn

    def _count_text(self, text: str) -> int:
        return self.count_fn([{"role": "user", "content": text}])

    def build_messages(
        self,
        system_identity: Dict,
        memory_messages: List[Dict],
        conversation_history: List[Dict],
    ) -> List[Dict]:
        """
        Membangun list pesan yang dikirim ke model dengan budget ketat.

        Args:
            system_identity: dict pesan system utama (identitas & kepribadian)
            memory_messages: list dict pesan memori yang akan diinjeksi
            conversation_history: list dict history percakapan (tanpa system)

        Returns:
            List pesan yang sudah dipotong sesuai budget
        """
        result = [system_identity]
        used_tokens = self.count_fn([system_identity])

        # --- Slot Memori ---
        memory_budget_left = self.budget.memory_budget
        trimmed_memories = []
        for mem_msg in memory_messages:
            cost = self.count_fn([mem_msg])
            if memory_budget_left - cost >= 0:
                trimmed_memories.append(mem_msg)
                memory_budget_left -= cost
            else:
                # Potong teks memori agar muat
                words = mem_msg["content"].split()
                while words and memory_budget_left < 0:
                    words = words[:-10]
                    cost = self.count_fn([{"role": mem_msg["role"],
                                           "content": " ".join(words)}])
                    memory_budget_left = self.budget.memory_budget - cost
                if words:
                    trimmed_memories.append({
                        "role": mem_msg["role"],
                        "content": " ".join(words) + "..."
                    })
                break  # sudah melebihi budget

        result.extend(trimmed_memories)
        used_tokens += self.count_fn(trimmed_memories) if trimmed_memories else 0

        # --- Slot Conversation History ---
        conv_budget = self.budget.available_total - used_tokens
        # Ambil dari yang paling baru, pertahankan urutan kronologis
        selected_conv = []
        for msg in reversed(conversation_history):
            cost = self.count_fn([msg])
            if conv_budget - cost >= 0:
                selected_conv.insert(0, msg)  # insert di depan agar urutan benar
                conv_budget -= cost
            else:
                break  # stop, budget habis

        result.extend(selected_conv)

        total = self.count_fn(result)
        return result, total
