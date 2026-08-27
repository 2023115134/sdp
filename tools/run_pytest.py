from __future__ import annotations

import pathlib
import runpy
import sys


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    deps = root / ".deps"

    sys.path.insert(0, str(root))
    if deps.exists():
        site = __import__("site")
        site.addsitedir(str(deps))

    import pytest

    raise SystemExit(pytest.main(["-q"]))


if __name__ == "__main__":
    main()
