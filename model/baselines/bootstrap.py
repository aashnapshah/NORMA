"""Put the pipeline's import roots on sys.path, from inside model/baselines/."""
import importlib.util
import os
import sys

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

_spec = importlib.util.spec_from_file_location(
    "_norma_import_roots_model", os.path.join(MODEL_DIR, "bootstrap.py"))
_roots = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_roots)
