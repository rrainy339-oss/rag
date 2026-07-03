from __future__ import annotations

import base64
import hashlib
import hmac
import secrets


DEFAULT_PASSWORD_ALGORITHM = "pbkdf2_sha256"
DEFAULT_PASSWORD_ITERATIONS = 260_000


def hash_password(
    password: str,
    *,
    iterations: int = DEFAULT_PASSWORD_ITERATIONS,
) -> str:
    salt = secrets.token_urlsafe(24)
    digest = _pbkdf2(password, salt=salt, iterations=iterations)
    return f"{DEFAULT_PASSWORD_ALGORITHM}${iterations}${salt}${digest}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations_text, salt, expected_digest = password_hash.split("$", 3)
        iterations = int(iterations_text)
    except ValueError:
        return False
    if algorithm != DEFAULT_PASSWORD_ALGORITHM:
        return False
    actual_digest = _pbkdf2(password, salt=salt, iterations=iterations)
    return hmac.compare_digest(actual_digest, expected_digest)


def _pbkdf2(password: str, *, salt: str, iterations: int) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    )
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
