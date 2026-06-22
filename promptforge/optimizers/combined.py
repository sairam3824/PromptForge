from __future__ import annotations

from typing import Callable

from ..example import Example
from ..module import Module
from ..tracker import RunTracker
from .bootstrap import BootstrapFewShot
from .instruction import InstructionSearch


class CombinedOptimizer:
    """Run InstructionSearch first, then BootstrapFewShot on the winning instruction."""

    def __init__(
        self,
        k: int = 3,
        n_proposals: int = 5,
        threshold: float = 0.5,
        tracker: RunTracker | None = None,
    ) -> None:
        self.k = k
        self.n_proposals = n_proposals
        self.threshold = threshold
        self.tracker = tracker

    def optimize(
        self,
        module: Module,
        train_examples: list[Example],
        val_examples: list[Example],
        metric: Callable[[str, str], float],
        run_id: str,
        budget: int = 60,
    ) -> Module:
        instr_budget = budget // 2
        boot_budget = budget - instr_budget

        best_instr = InstructionSearch(
            n_proposals=self.n_proposals,
            tracker=self.tracker,
        ).optimize(
            module=module,
            val_examples=val_examples,
            metric=metric,
            run_id=run_id,
            budget=instr_budget,
        )

        optimized = BootstrapFewShot(
            k=self.k,
            threshold=self.threshold,
            tracker=self.tracker,
        ).optimize(
            module=best_instr,
            train_examples=train_examples,
            metric=metric,
            run_id=run_id,
            budget=boot_budget,
            val_examples=val_examples,
        )

        return optimized
