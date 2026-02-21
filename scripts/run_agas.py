#!/usr/bin/env python
"""Convenience wrapper for `agas run-episode`."""

import sys

from agas.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "run-episode")
    raise SystemExit(main())
