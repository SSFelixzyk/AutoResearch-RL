# src/history.py
import json
from dataclasses import dataclass, field
from typing import List, Optional
from src.action_space import ArchSpec


@dataclass
class Experiment:
    spec: ArchSpec
    acc: float
    step_idx: int


class HistoryBuffer:
    def __init__(self, max_k: int = 10, top_k: int = 3):
        self.max_k = max_k
        self.top_k = top_k
        self.recent: List[Experiment] = []
        self.top_experiments: List[Experiment] = []   # sorted by acc desc, len <= top_k
        self.best_acc: float = 0.0
        self.best_spec: Optional[ArchSpec] = None
        self._step = 0

    def add(self, spec: ArchSpec, acc: float):
        exp = Experiment(spec=spec, acc=acc, step_idx=self._step)

        # sliding recent window
        self.recent.append(exp)
        if len(self.recent) > self.max_k:
            self.recent.pop(0)

        # top-k by accuracy
        if self.top_k > 0:
            self.top_experiments.append(exp)
            self.top_experiments.sort(key=lambda e: -e.acc)
            self.top_experiments = self.top_experiments[:self.top_k]

        if acc > self.best_acc:
            self.best_acc = acc
            self.best_spec = spec

        self._step += 1

    def build_prompt(self, program_md: str, use_history: bool) -> str:
        parts = [program_md.strip()]

        # history_k=0 means no history window — treat as use_history=False
        if use_history and self.max_k == 0:
            use_history = False

        if use_history and (self.recent or self.top_experiments):
            # Top-k best configs
            if self.top_experiments:
                parts.append("\n## Top Configurations (highest accuracy so far)\n")
                for rank, exp in enumerate(self.top_experiments, 1):
                    parts.append(f"#{rank} acc={exp.acc:.3f} | {exp.spec.to_summary()}")

            # Recent experiments
            if self.recent:
                parts.append("\n## Recent Experiments (most recent last)\n")
                recent_step_ids = {e.step_idx for e in self.recent}
                top_step_ids = {e.step_idx for e in self.top_experiments}
                for exp in self.recent:
                    marker = " ★" if exp.step_idx in top_step_ids else ""
                    parts.append(
                        f"Step {exp.step_idx}: acc={exp.acc:.3f}{marker} | {exp.spec.to_summary()}"
                    )

            parts.append(
                "\nPropose a configuration that differs meaningfully from the above. "
                "Vary at least one dimension (e.g. channels, blocks, norm, activation, "
                "optimizer, lr, augmentation) — do not copy any of the above configs."
            )

        elif use_history:
            parts.append("\n## Experiment History\n(No experiments yet — propose a starting config.)")

        parts.append("\n## Your Proposal\nRespond with valid JSON only:")
        return "\n".join(parts)
