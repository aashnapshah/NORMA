import torch
import pandas as pd
import datetime
import json
import pickle
from glob import glob
from pathlib import Path

def _default_labevents_usecols(all_columns):
  """Select a lean subset of columns commonly used from MIMIC labevents."""
  preferred = [
    "subject_id", "hadm_id", "itemid", "charttime", "specimen_id",
    "value", "valuenum", "valueuom", "flag"
  ]
  return [c for c in preferred if c in all_columns]

def save_data(f, data):
    print("Saving data to %s..." % f)
    if f.endswith(".csv"):
      data.to_csv(f, sep = "\t", index = False)
    elif f.endswith(".json"):
        with open(f, "w") as outfile:
          json.dump(data, outfile)
    elif f.endswith(".pth"):
        torch.save(data, f)
    else:
      raise NotImplementedError
    
def _default_labevents_dtypes(columns):
  """Lightweight dtypes to reduce memory; keep times as strings for speed."""
  dtypes = {}
  if "subject_id" in columns: dtypes["subject_id"] = "int32"
  if "hadm_id" in columns: dtypes["hadm_id"] = "Int32"  # nullable
  if "itemid" in columns: dtypes["itemid"] = "int32"
  if "specimen_id" in columns: dtypes["specimen_id"] = "Int64"  # nullable
  if "valuenum" in columns: dtypes["valuenum"] = "float32"
  if "value" in columns: dtypes["value"] = "string"
  if "valueuom" in columns: dtypes["valueuom"] = "category"
  if "flag" in columns: dtypes["flag"] = "category"
  # charttime left as string to avoid expensive parsing at load time
  return dtypes

def load_raw_data(data_dir, data_source = "MIMIC"):

  if data_source == "MIMIC":

    # Lab metadata
    print("Load raw metadata data at %s" %  (data_dir + "hosp/d_labitems.csv"))
    meta_df = pd.read_csv(data_dir + "hosp/d_labitems.csv")
    print(meta_df.head())
    print(meta_df.shape)
    print("Unique itemids:", len(meta_df["itemid"].unique()))
    print("Unique labels:", len(meta_df["label"].unique()))

    # Patient labs
    print("Load raw patient data at %s" % (data_dir + "hosp/labevents.csv"))
    pt_df = pd.read_csv(data_dir + "hosp/labevents.csv")
    print(pt_df.head())
    print(pt_df.shape)
    print(pt_df.columns)
    print("Unique patients:", len(pt_df["subject_id"].unique()))
    print("Unique labs (itemid):", len(pt_df["itemid"].unique()))

    # Combined patient labs with lab metadata
    pt_df_merged = pt_df.merge(meta_df, on="itemid", how="left")
    print("Merge raw meta and patient data")
    print(pt_df_merged.head())
    print(pt_df_merged.shape)
    return meta_df, pt_df, pt_df_merged

def load_raw_data_efficient(
  data_dir,
  data_source = "MIMIC",
  usecols = None,
  chunksize = 5_000_000,
  write_path = None,
  return_df = False
):
  """
  Memory-efficient loader for very large MIMIC lab datasets.

  - Streams labevents in chunks and joins to d_labitems on itemid using an indexed join.
  - Optionally writes merged output incrementally to Parquet (requires pyarrow) or CSV/CSV.GZ.
  - Set return_df=True ONLY if you know it fits in memory; otherwise stream to disk.

  Args:
    data_dir: base data directory containing hosp/d_labitems.csv and hosp/labevents.csv
    data_source: currently only "MIMIC" supported
    usecols: list of labevents columns to load; if None, a lean default is chosen
    chunksize: number of rows per chunk when reading labevents
    write_path: path to write merged output (e.g., ".../labevents_merged.parquet" or ".csv.gz"); if None and return_df=False, only prints summary stats
    return_df: if True, returns a concatenated DataFrame (may OOM for huge datasets)

  Returns:
    pandas.DataFrame if return_df=True, else None
  """
  if data_source != "MIMIC":
    raise ValueError("Only data_source='MIMIC' is supported by load_raw_data_efficient")

  data_dir = str(data_dir)
  d_labitems_path = str(Path(data_dir) / "hosp" / "d_labitems.csv")
  labevents_path = str(Path(data_dir) / "hosp" / "labevents.csv")

  print("[Efficient] Loading lab metadata:", d_labitems_path)
  meta_df = pd.read_csv(d_labitems_path)
  if "itemid" not in meta_df.columns:
    raise ValueError("d_labitems.csv must contain 'itemid'")
  # Tighten metadata dtypes and index for fast join
  meta_df["itemid"] = meta_df["itemid"].astype("int32", copy=False)
  meta_df = meta_df.set_index("itemid")

  # Discover columns and decide usecols/dtypes
  header = pd.read_csv(labevents_path, nrows=0)
  all_cols = header.columns.tolist()
  if usecols is None:
    usecols = _default_labevents_usecols(all_cols)
    if not usecols:
      # fallback: at least load itemid to allow join
      usecols = [c for c in ["subject_id", "hadm_id", "itemid"] if c in all_cols]
  dtypes = _default_labevents_dtypes(all_cols)

  print("[Efficient] Streaming labevents with columns:", usecols)
  print("[Efficient] Using dtypes:", {k: dtypes[k] for k in usecols if k in dtypes})

  # Optional writers
  parquet_writer = None
  csv_header_written = False
  rows_total = 0
  chunks_out = []

  try:
    reader = pd.read_csv(
      labevents_path,
      usecols=usecols,
      dtype=dtypes,
      chunksize=chunksize,
      low_memory=False
    )
  except ValueError:
    # If dtype conflicts occur, fallback to reading without dtype hints
    reader = pd.read_csv(
      labevents_path,
      usecols=usecols,
      chunksize=chunksize,
      low_memory=False
    )

  for i, chunk in enumerate(reader):
    # Ensure itemid type matches meta index
    if "itemid" in chunk.columns:
      chunk["itemid"] = chunk["itemid"].astype("int32", copy=False)

    # Join to metadata
    merged = chunk.join(meta_df, on="itemid", how="left", sort=False)

    # Optional: write out incrementally
    if write_path:
      if write_path.endswith(".parquet"):
        try:
          import pyarrow as pa  # noqa: F401
          import pyarrow.parquet as pq
          table = pa.Table.from_pandas(merged, preserve_index=False)
          if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(write_path, table.schema, compression="zstd")
          parquet_writer.write_table(table)
        except Exception as e:
          raise RuntimeError(f"Parquet write failed: {e}. Install 'pyarrow' or use a .csv/.csv.gz path.")
      else:
        # CSV or CSV.GZ append
        merged.to_csv(write_path, mode="a", header=not csv_header_written, index=False)
        csv_header_written = True

    if return_df:
      chunks_out.append(merged)

    rows_total += len(merged)
    if (i + 1) % 10 == 0:
      print(f"[Efficient] Processed {rows_total:,} rows...")

  # Close parquet writer if used
  if parquet_writer is not None:
    parquet_writer.close()

  print(f"[Efficient] Done. Total rows processed: {rows_total:,}")

  if return_df:
    print("[Efficient] Concatenating chunks into a single DataFrame (may use large memory)...")
    return pd.concat(chunks_out, ignore_index=True) if chunks_out else pd.DataFrame(columns=usecols)
  return None