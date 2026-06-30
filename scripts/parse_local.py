from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.domain.schemas import AccessControl
from app.ingestion.pipeline import LocalParsingPipeline
from app.parsing.base import ParserDependencyError
from app.parsing.docling_parser import SUPPORTED_PDF_BACKENDS


def main() -> int:
    args = _parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = LocalParsingPipeline(
        recursive=not args.no_recursive,
        pdf_backend=args.pdf_backend,
        do_ocr=args.ocr,
        artifacts_path=args.artifacts_path,
        access=_default_access(args),
        access_by_filename=_load_permissions(args.permissions) if args.permissions else None,
    )

    try:
        parsed_documents = list(pipeline.parse_path(args.input))
    except ParserDependencyError as exc:
        print(str(exc))
        return 2

    for document in parsed_documents:
        stem = Path(document.metadata.filename).stem
        json_path = output_dir / f"{stem}.parsed.json"
        markdown_path = output_dir / f"{stem}.parsed.md"
        json_path.write_text(document.to_json(), encoding="utf-8")
        markdown_path.write_text(document.markdown, encoding="utf-8")
        print(f"wrote {json_path}")
        print(f"wrote {markdown_path}")

    print(f"parsed {len(parsed_documents)} document(s)")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse local files with Docling.")
    parser.add_argument("--input", type=Path, required=True, help="File or folder to parse.")
    parser.add_argument("--output", type=Path, required=True, help="Output folder.")
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive discovery when input is a folder.",
    )
    parser.add_argument(
        "--pdf-backend",
        choices=sorted(SUPPORTED_PDF_BACKENDS),
        help=(
            "Docling PDF backend to use. Omit this to use Docling's default "
            "converter configuration."
        ),
    )
    parser.add_argument(
        "--ocr",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable Docling OCR for PDFs. Use --no-ocr for PDFs "
            "with an embedded text layer."
        ),
    )
    parser.add_argument(
        "--artifacts-path",
        type=Path,
        help=(
            "Local Docling model artifacts directory, for example D:\\AI\\docling. "
            "When set, Docling loads pre-downloaded models from this path."
        ),
    )
    parser.add_argument(
        "--tenant-id",
        help="Default tenant id written into parsed documents.",
    )
    parser.add_argument(
        "--allowed-user-id",
        dest="allowed_user_ids",
        action="append",
        default=[],
        help="Default allowed user id. Can be repeated.",
    )
    parser.add_argument(
        "--allowed-group-id",
        dest="allowed_group_ids",
        action="append",
        default=[],
        help="Default allowed group id. Can be repeated.",
    )
    parser.add_argument(
        "--classification",
        help="Default classification, for example public/internal/confidential.",
    )
    parser.add_argument(
        "--permissions",
        type=Path,
        help=(
            "JSON sidecar mapping filenames to access rules. Values use "
            "tenant_id, allowed_user_ids, allowed_group_ids, classification."
        ),
    )
    return parser.parse_args()


def _default_access(args: argparse.Namespace) -> AccessControl:
    return AccessControl(
        tenant_id=args.tenant_id,
        allowed_user_ids=args.allowed_user_ids,
        allowed_group_ids=args.allowed_group_ids,
        classification=args.classification,
    )


def _load_permissions(path: Path) -> dict[str, AccessControl]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "documents" in data:
        data = data["documents"]

    access_by_filename: dict[str, AccessControl] = {}
    if isinstance(data, dict):
        for filename, access_data in data.items():
            access_by_filename[str(filename)] = AccessControl.model_validate(access_data)
        return access_by_filename

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict) or "filename" not in item:
                raise ValueError("permission list items must include a filename field")
            filename = str(item["filename"])
            access_data = {key: value for key, value in item.items() if key != "filename"}
            access_by_filename[filename] = AccessControl.model_validate(access_data)
        return access_by_filename

    raise ValueError("permissions JSON must be a filename mapping or a documents list")


if __name__ == "__main__":
    raise SystemExit(main())
