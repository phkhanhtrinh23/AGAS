#!/usr/bin/env python
"""Convenience wrapper for `agas preprocess`."""

import sys

from agas.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "preprocess")
    raise SystemExit(main())
