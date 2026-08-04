class PipelineError(RuntimeError):
    """Base error for an expected, actionable pipeline failure."""


class ConfigError(PipelineError):
    """Raised when a pipeline configuration is invalid."""


class ExtractionError(PipelineError):
    """Raised when a source cannot be extracted safely."""

