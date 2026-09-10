#!/usr/bin/env python
"""Deprecated: use jobs/run_clalit.py.

This runner predates 04_refs (NORMA at the actual first-index time) and the
Cohen / Gaussian benchmarks, so it skipped stages 04, 05, 06, 14 and 16 and left
CHS on the reference intervals 01_process merged in from the server.  It forwards
to jobs/run_clalit.py, which runs the whole current pipeline chunk by chunk.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print(__doc__)
    print(f"forwarding: python jobs/run_clalit.py {' '.join(sys.argv[1:])}\n")
    sys.exit(subprocess.call([sys.executable, os.path.join(HERE, "run_clalit.py")] + sys.argv[1:]))
