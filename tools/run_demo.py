from __future__ import annotations

import pathlib
import runpy
import site
import sys


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    user_site = site.getusersitepackages()
    if user_site:
        site.addsitedir(user_site)

    runpy.run_module("app.demo", run_name="__main__")


if __name__ == "__main__":
    main()
