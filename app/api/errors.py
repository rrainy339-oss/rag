from __future__ import annotations


class RuntimeConfigurationError(RuntimeError):
    """Raised when the API runtime cannot be configured from current settings."""


class RuntimeUnavailableError(RuntimeError):
    """Raised when a configured runtime dependency is unavailable."""

