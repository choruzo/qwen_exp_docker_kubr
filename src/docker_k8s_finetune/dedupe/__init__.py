"""Exact and embedding-based dataset deduplication."""

from .exact import run_exact_dedupe
from .approximate import run_approximate_dedupe

__all__ = ["run_approximate_dedupe", "run_exact_dedupe"]
