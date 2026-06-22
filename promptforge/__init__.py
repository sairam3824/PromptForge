from .signature import Signature, Field
from .example import Example
from .module import Module
from .metrics import exact_match, f1_token, embedding_similarity, llm_judge, METRICS
from .cache import LLMCache
from .tracker import RunTracker
from .backends import OpenAIBackend, OllamaBackend
from .optimizers import BootstrapFewShot, InstructionSearch, CombinedOptimizer

__version__ = "0.1.0"

__all__ = [
    "Signature", "Field",
    "Example",
    "Module",
    "exact_match", "f1_token", "embedding_similarity", "llm_judge", "METRICS",
    "LLMCache",
    "RunTracker",
    "OpenAIBackend", "OllamaBackend",
    "BootstrapFewShot", "InstructionSearch", "CombinedOptimizer",
]
