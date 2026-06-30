from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.domain.permissions import (
    CLASSIFICATION_LEVEL_LTE_FILTER,
    PRINCIPAL_IDS_ANY_FILTER,
)


def metadata_matches(metadata: dict[str, Any], filters: dict[str, object]) -> bool:
    for key, expected in filters.items():
        if key == PRINCIPAL_IDS_ANY_FILTER:
            actual = metadata.get("principal_ids")
            if actual is None:
                continue
            if not _has_intersection(actual, expected):
                return False
            continue

        if key == CLASSIFICATION_LEVEL_LTE_FILTER:
            raw_actual = metadata.get("classification_level")
            if raw_actual is None:
                continue
            actual = _to_float(raw_actual)
            allowed = _to_float(expected)
            if actual is None or allowed is None or actual > allowed:
                return False
            continue

        if not _value_matches(metadata.get(key), expected):
            return False
    return True


def _value_matches(actual: object, expected: object) -> bool:
    if _is_many(expected):
        return _has_intersection(actual, expected)
    return actual == expected


def _has_intersection(actual: object, expected: object) -> bool:
    actual_values = set(_as_list(actual))
    expected_values = set(_as_list(expected))
    return bool(actual_values and expected_values and actual_values.intersection(expected_values))


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if _is_many(value):
        return list(value)
    return [value]


def _is_many(value: object) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict))


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
