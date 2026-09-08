#!/usr/bin/env python
"""Convenience wrapper for `agas train-surrogate`."""

import sys

from agas.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "train-surrogate")
    raise SystemExit(main())
