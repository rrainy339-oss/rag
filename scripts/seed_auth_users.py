from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.security.auth_service import default_seed_users  # noqa: E402
from app.security.config import SecuritySettings  # noqa: E402
from app.security.users import AuthUserStore  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed default auth users.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Auth SQLite database path. Defaults to RAG_AUTH_DB_PATH.",
    )
    args = parser.parse_args(argv)

    settings = SecuritySettings.from_env()
    db_path = args.db_path or settings.auth_db_path
    store = AuthUserStore(db_path)
    seed_users = default_seed_users(settings)
    store.seed_defaults(seed_users)
    print(f"seeded {len(seed_users)} auth users into {db_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
