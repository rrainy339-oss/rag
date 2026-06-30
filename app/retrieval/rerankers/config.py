from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class RerankerProvider(StrEnum):
    NOOP = "noop"
    BGE = "bge"
    QWEN = "qwen"
    COHERE = "cohere"
    VOYAGE = "voyage"


class RerankerConfig(BaseModel):
    provider: RerankerProvider = RerankerProvider.NOOP
    model_name: str | None = None
    batch_size: int = Field(default=16, ge=1)
    max_length: int | None = Field(default=None, ge=1)
    use_fp16: bool = False
    normalize: bool = False
    cache_dir: str | None = None
    device: str | None = None
    api_key: str | None = None
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_tokens_per_doc: int = Field(default=4096, ge=1)
    instruction: str | None = None
