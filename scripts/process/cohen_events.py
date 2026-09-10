#!/usr/bin/env python
"""Cohen et al. healthy-cohort criteria ("Approximation of health periods").

Their published exclusion lists, turned into per-source event tables and
applied to the dev cohorts (MIMIC-IV, EHRSHOT) at the lab-point level:

  Diagnosis of an analyte-linked condition -> that analyte is excluded for the
      patient from the first such diagnosis onward (and 6 months before)
  Paired medication -> the analyte is excluded for a lab point if the patient
      was exposed to a paired medication in the 6 months before to 1 month
      after the test
  Pregnancy ICD codes -> excluded during pregnancy (42-week window) and 30
      days after
  Hospitalization -> lab points during hospital admissions are excluded

This is cohort curation from raw records, not a reference-interval estimator,
so it sits in process/ with the other builders rather than in
model/baselines/cohen.py, which consumes load_events() and healthy_mask().
Was cohen_healthy.py, then the back half of cohen.py, until this split.

Build the event tables once with:  python process/cohen_events.py --build_healthy
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Static Cohen et al. inputs (ST2/ST3, the ICD-9->10 GEM, the RxNorm names and the
# medication->analyte map derived from them) are data/.  The per-source event tables
# built from raw MIMIC-IV / EHRSHOT records are raw results of those cohorts, so they
# live under results/raw/<source>/ like everything else a stage computes (2026-09-08).
CACHE_DIR = os.path.join(ROOT_DIR, "data", "cohen_healthy")
RAW_RESULTS_DIR = os.path.join(ROOT_DIR, "results", "raw")
EVENT_TABLES = ["dx_first", "preg", "hosp", "med"]


def event_path(source, name):
    """results/raw/<source>/cohen_<name>.parquet"""
    return os.path.join(RAW_RESULTS_DIR, source, f"cohen_{name}.parquet")

# Raw records, only needed by --build_healthy. Override with NORMA_RAW_DIR
# (or the two per-source variables) to point at your credentialed copies.
_RAW_DIR = os.environ.get("NORMA_RAW_DIR", os.path.join(os.path.dirname(ROOT_DIR), "data", "raw"))
RAW_MIMIC = os.environ.get("NORMA_RAW_MIMIC", os.path.join(_RAW_DIR, "mimiciv", "3.1", "hosp"))
RAW_EHRSHOT = os.environ.get("NORMA_RAW_EHRSHOT", os.path.join(_RAW_DIR, "ehrshot", "meds-omop-ehrshot"))

DX_LOOKAHEAD_DAYS = 183   # "in the past or in the next 6 months"
MED_BEFORE_DAYS = 183     # exposure 6 months before ...
MED_AFTER_DAYS = 30       # ... to 1 month after the lab test
PREG_BEFORE_DAYS = 294    # 42-week gestation window
PREG_AFTER_DAYS = 30      # "not pregnant in the last 30 d"

# Cohen et al. Methods: pregnancy and delivery ICD-9 codes; ICD-10 equivalents
PREG_ICD9_PREFIXES = (
    ["V22", "V23", "V27"] + [f"V3{i}" for i in range(8)]
    + [str(c) for c in range(633, 646)] + [str(c) for c in range(647, 677)]
)
PREG_ICD10_PREFIXES = ["O", "Z33", "Z34", "Z37", "Z3A"]

# ST3 lab names (Clalit) -> NORMA analyte codes. Unmapped ST3 labs are
# dropped; analytes without a mapped ST3 lab receive no medication filter.
ST3_LAB_TO_ANALYTE = {
    "ALBUMIN": "ALB",
    "ALT_Alanine_aminotransferase_GPT": "ALT",
    "AST_Aspartate_aminotransferase_GOT": "AST",
    "BILIRUBIN_DIRECT": "DBIL",
    "BILIRUBIN_TOTAL": "TBIL",
    "CALCIUM_BLOOD": "CA",
    "CHOLESTEROL": "TC",
    "CHOLESTEROL_HDL": "HDL",
    "CHOLESTEROL_LDL": "LDL",
    "CREATININE_BLOOD": "CRE",
    "C_REACTIVE_PROTEIN_CRP": "CRP",
    "Cl": "CL",
    "GAMMA_GLUTAMYL_TRANSPEPTIDASE": "GGT",
    "GLUCOSE_BLOOD": "GLU",
    "HCT": "HCT",
    "HEMOGLOBIN_A1C_CALCULATED": "A1C",
    "HGB": "HGB",
    "K": "K",
    "LACTIC_DEHYDROGENASE_LDH__BLOOD": "LDH",
    "MCH": "MCH",
    "MCHC": "MCHC",
    "MCV": "MCV",
    "MPV": "MPV",
    "Na": "NA",
    "PHOSPHATASE_ALKALINE": "ALP",
    "PLT": "PLT",
    "PROTEIN_TOTAL_BLOOD": "TP",
    "PT_SEC": "PT",
    "RBC": "RBC",
    "RDW": "RDW",
    "RDW_CV": "RDW",
    "TRIGLYCERIDES": "TGL",
    "UREA_BLOOD": "BUN",  # BUN ~ urea nitrogen; med associations carried over
    "WBC": "WBC",
}


def _norm_icd(code):
    return str(code).strip().upper().replace(".", "")


def load_exclusion_codes():
    """ST2 nonhealthy codes -> (icd9_prefixes, icd10_prefixes), GEM-expanded."""
    st2 = pd.read_csv(os.path.join(CACHE_DIR, "st2_nonhealthy_icd9.csv"),
                      dtype=str)
    codes = [_norm_icd(c) for c in st2["icd9"]]
    icd9 = sorted({c for c in codes if c[0].isdigit() or c[0] in "EV"})
    icd10 = sorted({c for c in codes if not (c[0].isdigit() or c[0] in "EV")})

    # Expand ICD-9 exclusions to ICD-10 via the CMS GEM crosswalk
    gem = pd.read_csv(os.path.join(CACHE_DIR, "icd9to10.csv"), dtype=str)
    gem["icd9cm"] = gem["icd9cm"].map(_norm_icd)
    gem["icd10cm"] = gem["icd10cm"].map(_norm_icd)
    gem = gem[gem["no_map"] != "1"]
    icd9_arr = gem["icd9cm"].values
    mapped = set()
    for p in icd9:
        hit = gem.loc[[c.startswith(p) for c in icd9_arr], "icd10cm"]
        mapped.update(hit.tolist())
    icd10 = sorted(set(icd10) | mapped)
    print(f"  ST2 exclusions: {len(icd9)} ICD-9 prefixes, "
          f"{len(icd10)} ICD-10 prefixes (GEM-expanded)")
    return icd9, icd10


def load_med_map():
    """ST3 -> {MEDNAME: set(analytes)} for meds paired with our analytes."""
    st3 = pd.read_csv(os.path.join(CACHE_DIR, "st3_lab_med_pairs.csv"))
    st3["analyte"] = st3["lab"].map(ST3_LAB_TO_ANALYTE)
    st3 = st3.dropna(subset=["analyte"])
    med_map = {}
    for med, grp in st3.groupby(st3["med"].str.upper().str.replace("_", " ")):
        med_map[med] = set(grp["analyte"])
    print(f"  ST3 med filter: {len(med_map)} meds across "
          f"{st3['analyte'].nunique()} analytes")
    return med_map


def _match_prefix(codes, prefixes):
    """Boolean mask: code starts with any prefix (prefix-grouped for speed)."""
    mask = np.zeros(len(codes), dtype=bool)
    arr = pd.Series(codes).astype(str).values
    for p in prefixes:
        mask |= np.char.startswith(arr.astype(str), p)
    return mask


# ── MIMIC-IV event tables ────────────────────────────────────────────────────

def build_mimic_events(icd9_px, icd10_px, med_map):
    adm = pd.read_csv(os.path.join(RAW_MIMIC, "admissions.csv"),
                      usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
                      parse_dates=["admittime", "dischtime"])

    dx = pd.read_csv(os.path.join(RAW_MIMIC, "diagnoses_icd.csv"),
                     usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
                     dtype={"icd_code": str})
    dx["icd_code"] = dx["icd_code"].map(_norm_icd)
    m9 = (dx["icd_version"] == 9) & _match_prefix(dx["icd_code"], icd9_px)
    m10 = (dx["icd_version"] == 10) & _match_prefix(dx["icd_code"], icd10_px)
    p9 = (dx["icd_version"] == 9) & _match_prefix(dx["icd_code"], PREG_ICD9_PREFIXES)
    p10 = (dx["icd_version"] == 10) & _match_prefix(dx["icd_code"], PREG_ICD10_PREFIXES)

    adm_t = adm.set_index("hadm_id")["admittime"]
    nonhealthy = dx[m9 | m10].copy()
    nonhealthy["date"] = nonhealthy["hadm_id"].map(adm_t)
    dx_first = nonhealthy.groupby("subject_id")["date"].min().reset_index()
    dx_first.columns = ["subject_id", "first_dx"]

    preg = dx[p9 | p10].copy()
    preg["date"] = preg["hadm_id"].map(adm_t)
    preg = preg[["subject_id", "date"]].dropna().drop_duplicates()

    hosp = adm[["subject_id", "admittime", "dischtime"]].dropna()

    # emar: medication exposures matched to ST3 meds (streamed)
    single = {m for m in med_map if " " not in m}
    multi = [m for m in med_map if " " in m]
    med_rows = []
    emar_path = os.path.join(RAW_MIMIC, "emar.csv.gz")
    for chunk in pd.read_csv(emar_path,
                             usecols=["subject_id", "charttime", "medication",
                                      "event_txt"],
                             chunksize=2_000_000):
        chunk = chunk[chunk["event_txt"].astype(str).str.contains(
            "Administer", case=False, na=False)]
        meds_u = chunk["medication"].astype(str).str.upper()
        tokens = meds_u.str.replace(r"[^A-Z ]", " ", regex=True).str.split()
        hit_med = []
        for toks, full in zip(tokens, meds_u):
            name = None
            for t in toks or []:
                if t in single:
                    name = t
                    break
            if name is None:
                for m in multi:
                    if m in full:
                        name = m
                        break
            hit_med.append(name)
        chunk = chunk.assign(med=hit_med).dropna(subset=["med"])
        med_rows.append(chunk[["subject_id", "charttime", "med"]])
    med = pd.concat(med_rows, ignore_index=True)
    med["charttime"] = pd.to_datetime(med["charttime"])
    med["date"] = med["charttime"].dt.floor("D")
    med = med[["subject_id", "date", "med"]].drop_duplicates()

    return {"dx_first": dx_first, "preg": preg, "hosp": hosp, "med": med}


# ── EHRSHOT event tables ─────────────────────────────────────────────────────

def build_ehrshot_events(icd9_px, icd10_px, med_map):
    # RxNorm CUI -> concatenated concept names (from NLM RxNorm prescribable
    # content; EHRSHOT's own metadata carries no RxNorm descriptions)
    rxn = pd.read_csv(os.path.join(CACHE_DIR, "rxnorm_names.csv"), dtype=str)
    rxn["STR"] = rxn["STR"].astype(str).str.upper()
    rx_desc = {f"RxNorm/{cui}": " ; ".join(g["STR"].unique())
               for cui, g in rxn.groupby("RXCUI")}

    single = {m for m in med_map if " " not in m}
    multi = [m for m in med_map if " " in m]

    dx_rows, preg_rows, med_rows, hosp_rows = [], [], [], []
    files = sorted(glob.glob(os.path.join(RAW_EHRSHOT, "data", "*.parquet")))
    for f in files:
        t = pd.read_parquet(f, columns=["subject_id", "time", "code", "end"])
        code = t["code"].astype(str)

        is_icd9 = code.str.startswith("ICD9CM/")
        is_icd10 = code.str.startswith("ICD10CM/")
        if is_icd9.any() or is_icd10.any():
            dxs = t[is_icd9 | is_icd10].copy()
            dxs["norm"] = dxs["code"].str.split("/").str[1].map(_norm_icd)
            v9 = dxs["code"].str.startswith("ICD9CM/")
            bad = np.zeros(len(dxs), dtype=bool)
            bad[v9.values] = _match_prefix(dxs.loc[v9, "norm"], icd9_px)
            bad[~v9.values] = _match_prefix(dxs.loc[~v9, "norm"], icd10_px)
            pr = np.zeros(len(dxs), dtype=bool)
            pr[v9.values] = _match_prefix(dxs.loc[v9, "norm"], PREG_ICD9_PREFIXES)
            pr[~v9.values] = _match_prefix(dxs.loc[~v9, "norm"], PREG_ICD10_PREFIXES)
            dx_rows.append(dxs.loc[bad, ["subject_id", "time"]])
            preg_rows.append(dxs.loc[pr, ["subject_id", "time"]])

        is_rx = code.str.startswith("RxNorm/")
        if is_rx.any():
            rx = t[is_rx].copy()
            desc = rx["code"].map(rx_desc).astype(str)
            toks = desc.str.replace(r"[^A-Z ]", " ", regex=True).str.split()
            hit = []
            for tk, full in zip(toks, desc):
                name = None
                for w in tk or []:
                    if w in single:
                        name = w
                        break
                if name is None:
                    for m in multi:
                        if m in full:
                            name = m
                            break
                hit.append(name)
            rx = rx.assign(med=hit).dropna(subset=["med"])
            med_rows.append(rx[["subject_id", "time", "med"]])

        is_ip = code.str.startswith("STANFORD_VISIT/Inpatient") | \
            code.str.startswith("Visit/IP") | code.str.startswith("Visit/ERIP")
        if is_ip.any():
            v = t[is_ip][["subject_id", "time", "end"]].copy()
            hosp_rows.append(v)

    dx = pd.concat(dx_rows, ignore_index=True)
    dx_first = dx.groupby("subject_id")["time"].min().reset_index()
    dx_first.columns = ["subject_id", "first_dx"]

    preg = pd.concat(preg_rows, ignore_index=True).rename(columns={"time": "date"})
    preg["date"] = pd.to_datetime(preg["date"]).dt.floor("D")
    preg = preg.drop_duplicates()

    med = pd.concat(med_rows, ignore_index=True)
    med["date"] = pd.to_datetime(med["time"]).dt.floor("D")
    med = med[["subject_id", "date", "med"]].drop_duplicates()

    hosp = pd.concat(hosp_rows, ignore_index=True)
    hosp["admittime"] = pd.to_datetime(hosp["time"])
    hosp["dischtime"] = pd.to_datetime(hosp["end"])
    # Visits without an end time: assume 1-day stay
    hosp["dischtime"] = hosp["dischtime"].fillna(hosp["admittime"] + pd.Timedelta(days=1))
    hosp = hosp[["subject_id", "admittime", "dischtime"]].dropna()

    return {"dx_first": dx_first, "preg": preg, "hosp": hosp, "med": med}


# ── Build / load ─────────────────────────────────────────────────────────────

def build_all(sources=("mimiciv", "ehrshot")):
    os.makedirs(CACHE_DIR, exist_ok=True)
    icd9_px, icd10_px = load_exclusion_codes()
    med_map = load_med_map()
    pd.Series({m: ",".join(sorted(a)) for m, a in med_map.items()}).rename(
        "analytes").rename_axis("med").reset_index().to_csv(
        os.path.join(CACHE_DIR, "med_analyte_map.csv"), index=False)

    builders = {"mimiciv": build_mimic_events, "ehrshot": build_ehrshot_events}
    for source in sources:
        print(f"  Building {source} event tables...")
        ev = builders[source](icd9_px, icd10_px, med_map)
        for name, df in ev.items():
            path = event_path(source, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_parquet(path, index=False)
            print(f"    {source} {name}: {len(df):,} rows -> {path}")


def load_events(source):
    ev = {}
    for name in EVENT_TABLES:
        df = pd.read_parquet(event_path(source, name))
        # parquet round-trips datetimes as [us]; merge_asof requires the same
        # unit as the [ns] point timestamps
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype("datetime64[ns]")
        ev[name] = df
    # keep_default_na: sodium's analyte code is the literal string "NA"
    med_map_df = pd.read_csv(os.path.join(CACHE_DIR, "med_analyte_map.csv"),
                             keep_default_na=False)
    ev["med_map"] = {r["med"]: set(r["analytes"].split(","))
                     for _, r in med_map_df.iterrows()}
    return ev


# ── Point-level healthy mask ─────────────────────────────────────────────────

def _asof_window_hit(points, events, by_cols, lo_offset_days, hi_offset_days):
    """True where an event exists within [t + lo, t + hi] for the point's key.

    events must have columns by_cols + ['date']. Vectorized via merge_asof:
    find the earliest event at/after (t + lo) and test it against (t + hi).
    """
    pts = points.reset_index(drop=True).copy()
    pts["_row"] = np.arange(len(pts))
    pts["_lo"] = pts["_t"] + pd.Timedelta(days=lo_offset_days)
    ev = events.dropna(subset=["date"]).sort_values("date")
    merged = pd.merge_asof(
        pts.sort_values("_lo"), ev, left_on="_lo", right_on="date",
        by=by_cols, direction="forward")
    merged = merged.sort_values("_row")
    hit = (merged["date"].notna()
           & (merged["date"].values
              <= (merged["_t"] + pd.Timedelta(days=hi_offset_days)).values))
    return hit.values


def healthy_mask(points, events):
    """Boolean mask over `points` (subject_id, analyte, time as datetime):
    True where the point falls in a Cohen-healthy context. Vectorized."""
    pts = points.reset_index(drop=True).copy()
    pts["_t"] = pd.to_datetime(pts["time"])
    n = len(pts)
    mask = np.ones(n, dtype=bool)

    # 1) Nonhealthy diagnosis in the past or next 6 months
    first_dx = events["dx_first"].set_index("subject_id")["first_dx"]
    fd = pts["subject_id"].map(first_dx)
    mask &= ~(fd.notna().values
              & (fd.values <= (pts["_t"] + pd.Timedelta(
                  days=DX_LOOKAHEAD_DAYS)).values))

    # 2) Pregnancy window: exclude if event in [t - 30d, t + 42wk]
    preg = events["preg"]
    if len(preg):
        mask &= ~_asof_window_hit(
            pts, preg[["subject_id", "date"]], ["subject_id"],
            -PREG_AFTER_DAYS, PREG_BEFORE_DAYS)

    # 3) Hospitalization periods: last admission at/before t still ongoing
    hosp = events["hosp"].dropna().sort_values("admittime")
    if len(hosp):
        pts["_row"] = np.arange(n)
        merged = pd.merge_asof(
            pts.sort_values("_t"), hosp, left_on="_t", right_on="admittime",
            by="subject_id", direction="backward")
        merged = merged.sort_values("_row")
        in_hosp = (merged["dischtime"].notna()
                   & (merged["_t"].values <= merged["dischtime"].values))
        mask &= ~in_hosp.values

    # 4) Paired-medication exposure in [t - 6mo, t + 1mo]
    med = events["med"]
    med_map = events["med_map"]
    if len(med):
        med = med.copy()
        med["analyte"] = med["med"].map(med_map)
        med = med.explode("analyte").dropna(subset=["analyte"])
        med = med[["subject_id", "analyte", "date"]].drop_duplicates()
        mask &= ~_asof_window_hit(
            pts, med, ["subject_id", "analyte"],
            -MED_BEFORE_DAYS, MED_AFTER_DAYS)

    return mask


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build_healthy", action="store_true",
                        help="build the healthy-cohort event tables from raw MIMIC-IV / EHRSHOT")
    parser.add_argument("--sources", type=str, nargs="*",
                        default=["mimiciv", "ehrshot"],
                        choices=["mimiciv", "ehrshot"])
    args = parser.parse_args()
    if args.build_healthy:
        build_all(sources=args.sources)
    else:
        parser.error("nothing to do: pass --build_healthy "
                     "(the models themselves are trained from 04_refs.py --only baselines)")
