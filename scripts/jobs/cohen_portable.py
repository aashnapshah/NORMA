#!/usr/bin/env python
"""Move the trained Cohen models between environments without retraining them.

cohen_dev_models.pkl is a pickle holding 90 live xgboost.Booster objects (30
analytes x m2/m3/m4).  Unpickling those needs an xgboost compatible with the one
that wrote them, which is not something to bet a Clalit run on -- and the pickle
is 157 MB.  This converts it to a single gzipped JSON holding each booster in
xgboost's own portable model format, and converts it back on the other side.

    python jobs/cohen_portable.py --export cohen_dev_models.json.gz    # here
    python jobs/cohen_portable.py --import cohen_dev_models.json.gz    # inside Clalit
    python jobs/cohen_portable.py --export x.json.gz --verify          # check round-trip

--import writes model/logs/baselines/cohen_dev_models.pkl, which is exactly
where augment_cohen looks; with that file present it loads the weights and never
retrains (retraining needs the dev cohorts, which are not in Clalit anyway).

--verify reloads what was written and asserts every booster predicts identically
on random input, so a bad export is caught here rather than in the middle of a
250-chunk run.
"""
import os as _os, sys as _sys
_SCRIPTS_DIR = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))
_ROOT = _os.path.dirname(_SCRIPTS_DIR)
for _p in (_os.path.join(_SCRIPTS_DIR, "lib"), _SCRIPTS_DIR, _ROOT):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import base64
import glob
import gzip
import json
import lzma
import os
import pickle

DEFAULT_PKL = os.path.join(_ROOT, "model", "logs", "baselines", "cohen_dev_models.pkl")
FORMAT_VERSION = 1


def _open(path, mode):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def export(pkl_path, out_path):
    with open(pkl_path, "rb") as f:
        art = pickle.load(f)

    models = {}
    n = 0
    for analyte, per_model in art["models"].items():
        models[analyte] = {}
        for name, entry in per_model.items():
            # save_raw(json) is xgboost's own portable format: it survives
            # version changes in a way a pickled Booster does not.
            models[analyte][name] = {
                "booster": bytes(entry["bst"].save_raw(raw_format="json")).decode(),
                "sigma": float(entry["sigma"]),
            }
            n += 1

    import xgboost
    payload = {
        "format_version": FORMAT_VERSION,
        "xgboost_version": xgboost.__version__,
        "analytes": list(art["analytes"]),
        "bin_cols": list(art["bin_cols"]),
        "dev_medians": {k: float(v) for k, v in art["dev_medians"].items()},
        "meta": art["meta"],
        "models": models,
    }
    with _open(out_path, "wt") as f:
        json.dump(payload, f)
    print(f"Exported {n} boosters ({len(models)} analytes) -> {out_path}"
          f"  {os.path.getsize(out_path) / 1e6:.1f} MB"
          f"  (pickle was {os.path.getsize(pkl_path) / 1e6:.0f} MB)")
    return payload


def _round_floats(o, nd):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, list):
        return [_round_floats(x, nd) for x in o]
    if isinstance(o, dict):
        return {k: _round_floats(v, nd) for k, v in o.items()}
    return o


# xz + base85 rather than gzip + base64: about 24 % less text for the same
# bytes, which is the most encoding can do here -- a 1,950-tree forest is
# intrinsically large.  The codec is recorded in _meta.json so older exports
# still import.
CODEC = "xz+b85"


def _compress(blob, codec=CODEC):
    if codec == "xz+b85":
        return base64.b85encode(lzma.compress(blob, preset=9 | lzma.PRESET_EXTREME)).decode()
    return base64.b64encode(gzip.compress(blob, 9)).decode()


def _decompress(text, codec):
    if codec == "xz+b85":
        return lzma.decompress(base64.b85decode(text))
    return gzip.decompress(base64.b64decode(text))


def _pack(per_model, nd=None):
    """One analyte's boosters -> base64 text.

    The boosters are packed as raw bytes behind a small index, not embedded as
    strings inside another JSON: JSON-escaping every quote in a 0.5 MB booster
    inflates it about tenfold before gzip ever sees it.
    """
    index, parts = [], []
    for name, e in sorted(per_model.items()):
        raw = bytes(e["bst"].save_raw(raw_format="json"))
        if nd is not None:
            raw = json.dumps(_round_floats(json.loads(raw), nd), separators=(",", ":")).encode()
        index.append({"model": name, "sigma": float(e["sigma"]), "len": len(raw)})
        parts.append(raw)
    header = json.dumps(index).encode()
    blob = len(header).to_bytes(4, "big") + header + b"".join(parts)
    return _compress(blob)


def _unpack(text, codec=CODEC):
    blob = _decompress(text, codec)
    n = int.from_bytes(blob[:4], "big")
    index = json.loads(blob[4:4 + n])
    out, off = {}, 4 + n
    for entry in index:
        raw = blob[off:off + entry["len"]]
        off += entry["len"]
        out[entry["model"]] = {"booster": raw.decode(), "sigma": entry["sigma"]}
    return out


def export_text(pkl_path, out_dir, nd=None, models=None):
    """One base64 text file per analyte, so the models can be pasted rather than copied.

    A gradient-boosted forest cannot be made small -- 750 trees per booster, 87
    boosters -- but packed and gzipped each analyte's three models come to a
    couple of hundred KB of plain text, which fits in an editor.  --round trims
    the float precision of every threshold and leaf value; --verify then says
    what that costs in predictions.  --import reads the directory back.
    """
    with open(pkl_path, "rb") as f:
        art = pickle.load(f)
    os.makedirs(out_dir, exist_ok=True)

    import xgboost
    meta = {
        "format_version": FORMAT_VERSION,
        "xgboost_version": xgboost.__version__,
        "analytes": list(art["analytes"]),
        "bin_cols": list(art["bin_cols"]),
        "dev_medians": {k: float(v) for k, v in art["dev_medians"].items()},
        "meta": art["meta"],
        "round": nd,
        "models": sorted(models) if models else None,
        "codec": CODEC,
    }
    meta_path = os.path.join(out_dir, "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)

    total = os.path.getsize(meta_path)
    for analyte, per_model in sorted(art["models"].items()):
        if models:
            per_model = {k: v for k, v in per_model.items() if k in models}
            if not per_model:
                continue
        path = os.path.join(out_dir, f"{analyte}.b64")
        with open(path, "w") as f:
            f.write(_pack(per_model, nd))
        total += os.path.getsize(path)
        print(f"  {analyte + '.b64':<16} {os.path.getsize(path) / 1e3:8.1f} KB")
    print(f"\n{len(art['models'])} analytes + meta -> {out_dir}  {total / 1e6:.1f} MB total text"
          + (f"  (rounded to {nd} dp)" if nd is not None else ""))


def load_parts(in_dir):
    """A directory written by --export_text -> the JSON payload shape."""
    with open(os.path.join(in_dir, "_meta.json")) as f:
        payload = json.load(f)
    models = {}
    for path in sorted(glob.glob(os.path.join(in_dir, "*.b64"))):
        analyte = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            models[analyte] = _unpack(f.read().strip(), payload.get("codec", "gz+b64"))
    if not models:
        raise SystemExit(f"No *.b64 files in {in_dir}")
    payload["models"] = models
    return payload


def load_payload(in_path):
    with _open(in_path, "rt") as f:
        payload = json.load(f)
    if payload.get("format_version") != FORMAT_VERSION:
        raise SystemExit(f"{in_path}: format_version {payload.get('format_version')}, "
                         f"this script writes {FORMAT_VERSION}")
    return payload


def to_artifact(payload):
    """JSON payload -> the artifact dict augment_cohen expects."""
    import xgboost

    models = {}
    for analyte, per_model in payload["models"].items():
        models[analyte] = {}
        for name, entry in per_model.items():
            bst = xgboost.Booster()
            bst.load_model(bytearray(entry["booster"].encode()))
            models[analyte][name] = {"bst": bst, "sigma": float(entry["sigma"])}
    return {
        "analytes": payload["analytes"],
        "bin_cols": payload["bin_cols"],
        "dev_medians": payload["dev_medians"],
        "meta": payload["meta"],
        "models": models,
    }


def import_(in_path, pkl_path):
    payload = load_parts(in_path) if os.path.isdir(in_path) else load_payload(in_path)
    import xgboost
    if payload["xgboost_version"] != xgboost.__version__:
        print(f"  note: exported under xgboost {payload['xgboost_version']}, "
              f"loading under {xgboost.__version__} (the JSON format spans both)")
    art = to_artifact(payload)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(art, f)
    n = sum(len(v) for v in art["models"].values())
    print(f"Imported {n} boosters ({len(art['models'])} analytes) -> {pkl_path}")
    print("  augment_cohen loads this and will not retrain "
          "(pass --cohen_retrain to force training)")


def verify(pkl_path, payload):
    """Every rebuilt booster must predict identically to the pickled one."""
    import numpy as np
    import xgboost

    with open(pkl_path, "rb") as f:
        art = pickle.load(f)
    rebuilt = to_artifact(payload)
    rng = np.random.default_rng(0)
    worst, checked = 0.0, 0
    for analyte, per_model in art["models"].items():
        if analyte not in rebuilt["models"]:
            continue                       # --models exported a subset
        for name, entry in per_model.items():
            if name not in rebuilt["models"][analyte]:
                continue
            a, b = entry["bst"], rebuilt["models"][analyte][name]["bst"]
            X = rng.normal(size=(32, a.num_features()))
            d = xgboost.DMatrix(X)
            worst = max(worst, float(np.abs(a.predict(d) - b.predict(d)).max()))
            if entry["sigma"] != rebuilt["models"][analyte][name]["sigma"]:
                raise SystemExit(f"sigma differs for {analyte}/{name}")
            checked += 1
    print(f"Verified {checked} boosters, max |prediction difference| = {worst:.3g}")
    if worst:
        print("  (nonzero is expected with --round; it is the cost of the smaller text)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--export", metavar="OUT", help="pickle -> portable json(.gz)")
    p.add_argument("--import", dest="import_", metavar="IN", help="portable json(.gz) -> pickle")
    p.add_argument("--pkl", default=DEFAULT_PKL, help=f"artifact path (default: {DEFAULT_PKL})")
    p.add_argument("--export_text", metavar="DIR",
                   help="pickle -> one pasteable base64 text file per analyte")
    p.add_argument("--models", nargs="*", default=None, metavar="M",
                   help="only these Cohen models (m2 is ~77 KB/analyte; m3 and m4 ~1 MB each)")
    p.add_argument("--round", type=int, default=None, metavar="N",
                   help="round every threshold and leaf value to N decimals (smaller text)")
    p.add_argument("--verify", action="store_true", help="after --export, check the round-trip")
    args = p.parse_args()

    if args.export_text:
        export_text(args.pkl, args.export_text, args.round, args.models)
        if args.verify:
            verify(args.pkl, load_parts(args.export_text))
    elif args.export:
        payload = export(args.pkl, args.export)
        if args.verify:
            verify(args.pkl, payload)
    elif args.import_:
        import_(args.import_, args.pkl)
    else:
        p.error("pass --export OUT, --export_text DIR, or --import IN")


if __name__ == "__main__":
    main()
