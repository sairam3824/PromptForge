from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Callable

MetricFn = Callable[[Any, Any], float]


def _normalize(s: str) -> str:
    s = str(s).lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def exact_match(prediction: Any, gold: Any) -> float:
    return 1.0 if _normalize(prediction) == _normalize(gold) else 0.0


def f1_token(prediction: Any, gold: Any) -> float:
    """Token-level F1 (SQuAD-style)."""
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


_st_cache: dict[str, Any] = {}


def embedding_similarity(
    prediction: Any,
    gold: Any,
    model_name: str = "all-MiniLM-L6-v2",
) -> float:
    """Cosine similarity of sentence embeddings. Requires: pip install sentence-transformers"""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        raise ImportError(
            "embedding_similarity requires sentence-transformers:\n"
            "  pip install 'promptforge[embeddings]'"
        )
    if model_name not in _st_cache:
        _st_cache[model_name] = SentenceTransformer(model_name)
    model = _st_cache[model_name]
    embs = model.encode([str(prediction), str(gold)])
    cos = float(
        np.dot(embs[0], embs[1])
        / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-9)
    )
    return max(0.0, cos)


_JUDGE_SYSTEM = (
    "You are an impartial evaluator. "
    "Rate how correct the predicted answer is compared to the gold answer. "
    "Respond with ONLY a number from 0.0 (completely wrong) to 1.0 (perfectly correct)."
)


def llm_judge(
    prediction: Any,
    gold: Any,
    backend=None,
    task_description: str = "",
) -> float:
    """Score prediction vs gold using an LLM. Returns 0.0–1.0."""
    if backend is None:
        from .backends.openai_backend import OpenAIBackend

        backend = OpenAIBackend()

    ctx = f"Task context: {task_description}\n\n" if task_description else ""
    prompt = f"{ctx}Predicted: {prediction}\nGold: {gold}\nScore (0.0–1.0):"
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    response = backend.generate(messages)
    try:
        score = float(re.search(r"[\d.]+", response).group())  # type: ignore[union-attr]
        return min(1.0, max(0.0, score))
    except (AttributeError, ValueError):
        return 0.0


METRICS: dict[str, MetricFn] = {
    "exact_match": exact_match,
    "f1": f1_token,
    "embedding": embedding_similarity,
    "llm_judge": llm_judge,
}
