from __future__ import annotations

import random
from typing import Callable

from ..example import Example
from ..module import Module
from ..tracker import RunTracker


class BootstrapFewShot:
    """
    Run the module on training examples, collect those where metric >= threshold,
    then inject the best k as few-shot demos.
    """

    def __init__(
        self,
        k: int = 3,
        threshold: float = 0.5,
        max_pool: int = 50,
        tracker: RunTracker | None = None,
    ) -> None:
        self.k = k
        self.threshold = threshold
        self.max_pool = max_pool
        self.tracker = tracker

    def _score_all(
        self,
        module: Module,
        examples: list[Example],
        metric: Callable,
        output_field: str,
        budget: int,
    ) -> list[tuple[float, Example]]:
        scored: list[tuple[float, Example]] = []
        start_count = module.backend.call_count
        for ex in examples:
            if (module.backend.call_count - start_count) >= budget:
                break
            try:
                pred = module(**ex.inputs)
                score = metric(pred.get(output_field, ""), ex.outputs.get(output_field, ""))
            except Exception:
                score = 0.0
            if score >= self.threshold:
                scored.append((score, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    @staticmethod
    def _evaluate(
        module: Module,
        examples: list[Example],
        metric: Callable,
        output_field: str,
    ) -> float:
        if not examples:
            return 0.0
        scores = []
        for ex in examples:
            try:
                pred = module(**ex.inputs)
                scores.append(metric(pred.get(output_field, ""), ex.outputs.get(output_field, "")))
            except Exception:
                scores.append(0.0)
        return sum(scores) / len(scores)

    def optimize(
        self,
        module: Module,
        train_examples: list[Example],
        metric: Callable[[str, str], float],
        run_id: str,
        budget: int = 50,
        val_examples: list[Example] | None = None,
    ) -> Module:
        output_field = module.signature.outputs[0].name
        scored = self._score_all(module, train_examples, metric, output_field, budget)

        pool = [ex for _, ex in scored[: self.max_pool]]
        if len(pool) > self.k:
            random.shuffle(pool)
            demos = pool[: self.k]
        else:
            demos = pool

        optimized = module.clone(demos=demos)

        val_score = 0.0
        if val_examples:
            val_score = self._evaluate(optimized, val_examples, metric, output_field)

        if self.tracker:
            self.tracker.log(
                run_id=run_id,
                optimizer="BootstrapFewShot",
                iteration=0,
                instruction=optimized.instruction,
                demos=[ex.to_dict() for ex in demos],
                val_score=val_score,
            )

        return optimized
