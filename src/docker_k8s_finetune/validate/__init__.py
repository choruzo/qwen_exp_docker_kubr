"""Syntactic validation for generated Kubernetes YAML and Dockerfiles."""

from .pipeline import run_syntax_validation

__all__ = ["run_syntax_validation"]
