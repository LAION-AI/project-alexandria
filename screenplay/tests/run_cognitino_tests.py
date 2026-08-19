#!/usr/bin/env python3
"""Dependency-free runner for the abstraction-layer negative-case suite.

pytest is not installed on the extraction host, and a test suite that cannot run where the
pipeline runs is a suite that will not be run. Exits non-zero on any failure, and prints
the list of check ids whose negative cases passed — that list is what ``run_all`` requires
before it will report any check as ``pass``.
"""

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_cognitino_checks as suite  # noqa: E402


def main() -> int:
    tests = sorted(name for name in dir(suite) if name.startswith("test_"))
    failures = []
    for name in tests:
        try:
            getattr(suite, name)()
            print("  pass  {}".format(name))
        except Exception:
            failures.append(name)
            print("  FAIL  {}".format(name))
            traceback.print_exc()

    covered = sorted({
        match.group(1).upper()
        for name in tests if name not in failures
        for match in [re.match(r"test_(g\d)_", name)] if match
    })
    print("\n{} passed, {} failed".format(len(tests) - len(failures), len(failures)))
    print("negative cases verified for: {}".format(", ".join(covered) or "none"))
    if failures:
        print("FAILED: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
