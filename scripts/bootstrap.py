"""Put the pipeline's import roots on sys.path."""
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.dirname(SCRIPTS_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "model")

for _p in (os.path.join(SCRIPTS_DIR, "lib"), SCRIPTS_DIR,
           os.path.join(SCRIPTS_DIR, "process"), BASE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
