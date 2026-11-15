"""Regenerates the committed synthetic fixtures under tests/fixtures/synthetic.

Run manually from the repository root:

    python tests/generate_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from fixtures_helpers import build_synthetic_dataset  # noqa: E402


def main() -> None:
    target = REPO_ROOT / "tests" / "fixtures" / "synthetic"
    build_synthetic_dataset(target)
    print(f"Synthetic fixtures regenerated at {target}")


if __name__ == "__main__":
    main()
