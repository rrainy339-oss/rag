from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from app.security.schemas import AuthError


def encode_jwt(
    claims: dict[str, Any],
    *,
    secret: str,
    expires_in_seconds: int,
    issuer: str | None = None,
    audience: str | None = None,
) -> str:
    now = int(time.time())
    payload = dict(claims)
    payload["iat"] = now
    payload["exp"] = now + expires_in_seconds
    if issuer:
        payload["iss"] = issuer
    if audience:
        payload["aud"] = audience

    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = ".".join(
        [
            _base64url_json(header),
            _base64url_json(payload),
        ]
    )
    signature = _sign(signing_input, secret)
    return f"{signing_input}.{signature}"


def decode_jwt(
    token: str,
    *,
    secret: str,
    issuer: str | None = None,
    audience: str | None = None,
) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthError("Invalid JWT format.", status_code=401)

    header = _decode_json_part(parts[0])
    if header.get("alg") != "HS256":
        raise AuthError("Unsupported JWT algorithm.", status_code=401)

    signing_input = f"{parts[0]}.{parts[1]}"
    expected_signature = _sign(signing_input, secret)
    if not hmac.compare_digest(expected_signature, parts[2]):
        raise AuthError("Invalid JWT signature.", status_code=401)

    claims = _decode_json_part(parts[1])
    now = int(time.time())
    exp = _int_claim(claims.get("exp"))
    if exp is None or exp <= now:
        raise AuthError("JWT has expired.", status_code=401)

    nbf = _int_claim(claims.get("nbf"))
    if nbf is not None and nbf > now:
        raise AuthError("JWT is not active yet.", status_code=401)

    if issuer and claims.get("iss") != issuer:
        raise AuthError("Invalid JWT issuer.", status_code=401)
    if audience and claims.get("aud") != audience:
        raise AuthError("Invalid JWT audience.", status_code=401)

    return claims


def _base64url_json(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _base64url(raw)


def _decode_json_part(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(_base64url_decode(value).decode("utf-8"))
    except Exception as exc:
        raise AuthError("Invalid JWT payload.", status_code=401) from exc
    if not isinstance(decoded, dict):
        raise AuthError("Invalid JWT payload.", status_code=401)
    return decoded


def _sign(signing_input: str, secret: str) -> str:
    digest = hmac.new(
        secret.encode("utf-8"),
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return _base64url(digest)


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}")


def _int_claim(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
