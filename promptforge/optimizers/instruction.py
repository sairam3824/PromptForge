from __future__ import annotations

from typing import Callable

from ..example import Example
from ..module import Module
from ..tracker import RunTracker

_PROPOSAL_SYSTEM = (
    "You are a prompt engineering expert. "
    "Generate alternative phrasings of the given task instruction. "
    "Vary the style, specificity, and framing to improve task performance."
)


class InstructionSearch:
    """
    Propose N alternative instructions via the LLM, evaluate each on the val split,
    keep the best (greedy hill-climb within a call budget).
    """

    def __init__(
        self,
        n_proposals: int = 5,
        tracker: RunTracker | None = None,
    ) -> None:
        self.n_proposals = n_proposals
        self.tracker = tracker

    def _propose(self, module: Module, current_instruction: str, n: int) -> list[str]:
        prompt = (
            f"Current instruction:\n{current_instruction}\n\n"
            f"Generate {n} alternative phrasings. "
            "Each should clearly convey the same task but with different wording.\n"
            "Return ONLY a numbered list:\n1. <instruction>\n2. <instruction>\n..."
        )
        messages = [
            {"role": "system", "content": _PROPOSAL_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        response = module.backend.generate(messages)
        proposals: list[str] = []
        for line in response.strip().splitlines():
            line = line.strip()
            if line and line[0].isdigit() and ". " in line:
                proposals.append(line.split(". ", 1)[1].strip())
        return proposals[:n]

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
        val_examples: list[Example],
        metric: Callable[[str, str], float],
        run_id: str,
        budget: int = 30,
    ) -> Module:
        if not val_examples:
            return module

        output_field = module.signature.outputs[0].name
        best_module = module
        best_score = self._evaluate(module, val_examples, metric, output_field)
        calls_used = len(val_examples)

        if self.tracker:
            self.tracker.log(
                run_id=run_id,
                optimizer="InstructionSearch",
                iteration=0,
                instruction=module.instruction,
                demos=[ex.to_dict() for ex in module.demos],
                val_score=best_score,
                metadata={"status": "baseline"},
            )

        iteration = 1
        remaining = budget - calls_used

        while remaining > len(val_examples):  # need room for proposal call + at least one eval
            proposals = self._propose(best_module, best_module.instruction, self.n_proposals)
            remaining -= 1  # one call for proposals

            for proposal in proposals:
                if remaining < len(val_examples):
                    break
                candidate = best_module.clone(instruction=proposal)
                score = self._evaluate(candidate, val_examples, metric, output_field)
                remaining -= len(val_examples)

                if self.tracker:
                    self.tracker.log(
                        run_id=run_id,
                        optimizer="InstructionSearch",
                        iteration=iteration,
                        instruction=proposal,
                        demos=[ex.to_dict() for ex in candidate.demos],
                        val_score=score,
                    )

                if score > best_score:
                    best_score = score
                    best_module = candidate

                iteration += 1

        return best_module
