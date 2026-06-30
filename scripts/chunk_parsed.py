from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunking.config import ChunkingConfig
from app.chunking.pipeline import ChunkingPipeline
from app.domain.schemas import ParsedDocument


def main() -> int:
    args = _parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ChunkingPipeline(
        ChunkingConfig(
            child_target_tokens=args.child_target_tokens,
            child_max_tokens=args.child_max_tokens,
            parent_target_tokens=args.parent_target_tokens,
            parent_max_tokens=args.parent_max_tokens,
        )
    )

    input_files = _iter_input_files(args.input)
    for input_file in input_files:
        document = ParsedDocument.model_validate_json(
            input_file.read_text(encoding="utf-8")
        )
        result = pipeline.chunk_document(document)
        output_path = output_dir / input_file.name.replace(".parsed.json", ".chunks.json")
        if output_path == output_dir / input_file.name:
            output_path = output_dir / f"{input_file.stem}.chunks.json"
        output_path.write_text(result.to_json(), encoding="utf-8")
        print(f"wrote {output_path}")

    print(f"chunked {len(input_files)} parsed document(s)")
    return 0


def _iter_input_files(path: Path) -> list[Path]:
    resolved = path.resolve()
    if resolved.is_file():
        return [resolved]
    return sorted(resolved.glob("*.parsed.json"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk normalized ParsedDocument JSON.")
    parser.add_argument("--input", type=Path, required=True, help="Parsed JSON file or folder.")
    parser.add_argument("--output", type=Path, required=True, help="Output folder.")
    parser.add_argument("--child-target-tokens", type=int, default=450)
    parser.add_argument("--child-max-tokens", type=int, default=650)
    parser.add_argument("--parent-target-tokens", type=int, default=1400)
    parser.add_argument("--parent-max-tokens", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

