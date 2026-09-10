"""Put the pipeline's import roots on sys.path.

Every stage script starts with

    import bootstrap  # noqa: F401

instead of repeating the path arithmetic.  Python always puts the running
script's own folder (norma/scripts) on sys.path, so this is importable from any
working directory, and it adds:

    scripts/lib       constants, datasets, figlib, metrics, models
    scripts           the process package (process.config)
    scripts/process   the bare `from config import ...` that process/ uses internally
    norma             model/ -- the NORMA code and its baselines

Sixteen copies of this used to live at the top of the stage scripts, each with
its own guess at the layout; the copies under format/ still pointed at the old
removed validation/ tree, which is why the figure build stopped importing.  One copy
means moving the tree is one edit.
"""
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.dirname(SCRIPTS_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "model")

for _p in (os.path.join(SCRIPTS_DIR, "lib"), SCRIPTS_DIR,
           os.path.join(SCRIPTS_DIR, "process"), BASE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
