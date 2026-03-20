"""
Offline script: pre-computes the example cache and saves it to
data/example_cache.json so the app can run on Render without the
1.4 GB sequences file.

Run once locally:  python build_cache.py
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'process'))

from dotenv import load_dotenv
load_dotenv()

# Trigger the same startup as app.py
import app as _app

out_path = os.path.join(os.path.dirname(__file__), 'data', 'example_cache.json')
os.makedirs(os.path.dirname(out_path), exist_ok=True)

cache = _app._EXAMPLE_CACHE
json.dump(cache, open(out_path, 'w'), indent=None)

total = sum(len(v) for v in cache.values())
print(f"\nSaved {total} examples across {len(cache)} tests to {out_path}")
print(f"File size: {os.path.getsize(out_path) / 1024:.0f} KB")
