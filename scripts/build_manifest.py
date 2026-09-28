#!/usr/bin/env python3
"""Write a SHA-256 manifest for the release."""
from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "metadata" / "MANIFEST.sha256"
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path != output
        and "__pycache__" not in path.parts
        and ".venv" not in path.parts
    )
    lines = []
    for path in paths:
        digest = file_sha256(path)
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    output.write_text("\n".join(lines) + "\n")
    print(f"[manifest] wrote {len(lines)} checksums")


if __name__ == "__main__":
    main()
