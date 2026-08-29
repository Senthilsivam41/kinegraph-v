"""RAG evaluation package with side-effect-free submodule imports.

The evaluator requires production settings (API keys and Neo4j credentials),
but data-only helpers such as ``eval.regression_gate`` must be usable by CI
before those settings are loaded.  Keep the public evaluator names available
through lazy attribute loading instead of importing the live evaluator at
package import time.
"""

__all__ = [
    "RAGASConfigurationError",
    "RAGASEvaluator",
    "RAGASValidationError",
    "require_successful_ragas",
]


def __getattr__(name):
    if name in __all__:
        from eval import ragas_evaluator

        return getattr(ragas_evaluator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
