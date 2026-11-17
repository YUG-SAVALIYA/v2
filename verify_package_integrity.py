"""Verify every packaged file against SHA256SUMS.txt."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    sums = ROOT / "SHA256SUMS.txt"
    if not sums.exists():
        raise FileNotFoundError(f"Missing integrity manifest: {sums}")
    failures: list[str] = []
    entries = 0
    for line in sums.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        target = ROOT / Path(relative)
        entries += 1
        if not target.exists():
            failures.append(f"missing: {relative}")
        elif digest(target) != expected:
            failures.append(f"hash mismatch: {relative}")
    if failures:
        print("PACKAGE INTEGRITY CHECK FAILED")
        for failure in failures:
            print("-", failure)
        raise SystemExit(1)
    print(f"PACKAGE INTEGRITY CHECK PASSED: {entries} files match SHA256SUMS.txt")


if __name__ == "__main__":
    main()
