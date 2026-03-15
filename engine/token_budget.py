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
    def __init__(self, budget: TokenBudget, count_fn):
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
        result = [system_identity]
        used_tokens = self.count_fn([system_identity])
        memory_budget_left = self.budget.memory_budget
        trimmed_memories = []
        for mem_msg in memory_messages:
            cost = self.count_fn([mem_msg])
            if memory_budget_left - cost >= 0:
                trimmed_memories.append(mem_msg)
                memory_budget_left -= cost
            else:
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
                break

        result.extend(trimmed_memories)
        used_tokens += self.count_fn(trimmed_memories) if trimmed_memories else 0

        conv_budget = self.budget.available_total - used_tokens
        selected_conv = []
        for msg in reversed(conversation_history):
            cost = self.count_fn([msg])
            if conv_budget - cost >= 0:
                selected_conv.insert(0, msg)
                conv_budget -= cost
            else:
                break

        result.extend(selected_conv)

        total = self.count_fn(result)
        return result, total
