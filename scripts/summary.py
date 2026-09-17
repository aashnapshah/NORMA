#!/usr/bin/env python
"""Bind every figure and table PDF in results/{figures,tables}/all into one document.

Each source page is included whole, captioned with its own title (the stage number and the
file's name), so the summary is a browsable index of the pipeline's output rather than a
second rendering of it.

    python scripts/summary.py               # -> results/summary.pdf
"""
import argparse
import os
import re
import shutil
import subprocess
import tempfile

import bootstrap  # noqa: F401
from lib.figlib import BASE_DIR, _tectonic

RESULTS = os.path.join(BASE_DIR, "results")
SOURCES = [("Figure", os.path.join(RESULTS, "figures", "all")),
           ("Table", os.path.join(RESULTS, "tables", "all", "pdf"))]

# Words that are not capitalised mid-title, and ones that are not merely Title Case.
SMALL = {"of", "in", "by", "to", "at", "and", "vs", "per", "for"}
ACRONYM = {"ri": "RI", "auroc": "AUROC", "nri": "NRI", "cox": "Cox", "km": "KM", "roc": "ROC",
           "chs": "CHS", "eicu": "eICU", "inspire": "INSPIRE", "mae": "MAE", "mape": "MAPE",
           "r2": "R2", "ckd": "CKD", "aki": "AKI", "t2d": "T2D", "los": "LOS", "icu": "ICU",
           "norma": "NORMA", "nig": "NIG", "circos": "circos"}


def _title(kind, stem):
    """'11_lead_time_analyte' -> 'Figure 11 | Lead Time Analyte'."""
    stage, _, rest = stem.partition("_")
    if not stage.isdigit():
        stage, rest = "", stem
    words = []
    for i, w in enumerate(rest.split("_")):
        if w.lower() in ACRONYM:
            words.append(ACRONYM[w.lower()])
        elif i and w.lower() in SMALL:
            words.append(w.lower())
        else:
            words.append(w.capitalize())
    head = f"{kind} {stage}" if stage else kind
    return f"{head} | {' '.join(words)}"


def _tex_escape(s):
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("|", r"$|$")


def _bbox(path, page):
    """(trim-left, bottom, right, top) in bp that crops a page to its ink, or None."""
    r = subprocess.run(["gs", "-dNOPAUSE", "-dBATCH", "-q", "-sDEVICE=bbox",
                        f"-dFirstPage={page}", f"-dLastPage={page}", path],
                       capture_output=True, text=True)
    box = media = None
    for line in (r.stderr or "").splitlines():
        if line.startswith("%%HiResBoundingBox:"):
            box = [float(x) for x in line.split()[1:5]]
    out = subprocess.run(["pdfinfo", "-f", str(page), "-l", str(page), path],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        m = re.match(r"Page\s+\d+ size: ([\d.]+) x ([\d.]+) pts", line)
        if m:
            media = (float(m.group(1)), float(m.group(2)))
            break
    if not box or not media:
        return None
    pad = 6.0
    w, h = media
    trim = (max(box[0] - pad, 0), max(box[1] - pad, 0),
            max(w - box[2] - pad, 0), max(h - box[3] - pad, 0))
    return trim if all(t >= 0 for t in trim) and box[2] > box[0] and box[3] > box[1] else None


def _page_count(path):
    out = subprocess.run(["pdfinfo", path], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split()[1])
    return 1


def build(out_path):
    tectonic = _tectonic()
    if tectonic is None:
        raise SystemExit("tectonic not found; cannot build the summary")
    entries = []
    for kind, d in SOURCES:
        if not os.path.isdir(d):
            print(f"  [skip] {os.path.relpath(d, BASE_DIR)} missing")
            continue
        for f in sorted(os.listdir(d)):
            # ._name.pdf are macOS resource forks copied in alongside the real file.
            if f.endswith(".pdf") and not f.startswith("._"):
                entries.append((kind, os.path.join(d, f), _title(kind, f[:-4])))
    if not entries:
        raise SystemExit("no source PDFs found")

    # One output page per source page: the artwork scaled to fit above its title.  \includepdf
    # would paint the source page over the whole sheet, leaving nowhere for a caption.
    body = []
    for kind, path, title in entries:
        for n in range(1, _page_count(path) + 1):
            trim = _bbox(path, n)
            crop = (r",trim=%.1fbp %.1fbp %.1fbp %.1fbp,clip" % trim) if trim else ""
            body.append(r"\begin{center}" "\n"
                        r"\includegraphics[page=" + str(n) + crop +
                        r",width=\textwidth,height=0.93\textheight,keepaspectratio]{"
                        + path + "}\\\\[8pt]\n"
                        r"{\small " + _tex_escape(title) + r"}" "\n" r"\end{center}" "\n"
                        r"\clearpage")
    tex = ("\\documentclass[11pt]{article}\n"
           "\\usepackage[margin=0.4in]{geometry}\n"
           "\\usepackage{graphicx}\n\\usepackage{lmodern}\n"
           "\\renewcommand{\\familydefault}{\\sfdefault}\n"
           "\\setlength{\\parindent}{0pt}\n\\pagestyle{empty}\n\\begin{document}\n"
           + "\n".join(body) + "\n\\end{document}\n")

    work = tempfile.mkdtemp(prefix="summary_")
    src = os.path.join(work, "summary.tex")
    with open(src, "w") as f:
        f.write(tex)
    r = subprocess.run([tectonic, src], capture_output=True, text=True, cwd=work)
    if r.returncode != 0:
        raise SystemExit(f"LaTeX failed:\n{r.stderr.strip()[-2000:]}")
    shutil.move(src.replace(".tex", ".pdf"), out_path)
    shutil.rmtree(work, ignore_errors=True)
    print(f"  {len(entries)} source PDFs -> {os.path.relpath(out_path, BASE_DIR)}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default=os.path.join(RESULTS, "summary.pdf"))
    build(p.parse_args().out)


if __name__ == "__main__":
    main()
