"""Put the pipeline's import roots on sys.path, from inside model/.

Python always puts the running script's own folder on sys.path first, so a bare

    import bootstrap  # noqa: F401

resolves to this file for anything in model/ and to scripts/bootstrap.py for
the stage scripts. Either way the roots come from the one list in
scripts/bootstrap.py, which this file loads by path (importing it by name would
find this module again).

Nine copies of that path arithmetic used to live in model/, and eight of them
still pointed at the top-level process/ that moved under scripts/ on
2026-09-08 -- which is why model/data.py, priors.py, shrinkage.py,
state_prior.py, sensitivity_analysis.py and predict_states.py could only be
imported by a caller that had already fixed sys.path for them.
"""
import importlib.util
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
MODEL_DIR = os.path.join(BASE_DIR, "model")

if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

_spec = importlib.util.spec_from_file_location(
    "_norma_import_roots", os.path.join(SCRIPTS_DIR, "bootstrap.py"))
_roots = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_roots)
