"""
Generate an interactive, self-contained HTML web-summary report for a GeneExt run.

Layout inspired by the CellRanger web_summary:
  - Stat cards row  (genes extended, median / mean / max extension, peak counts)
  - Extension-length distribution chart  (histogram, interactive)
  - Peak-coverage distribution chart     (log10 overlay, genic vs non-genic)
  - Mapping-stats comparison table       (only when --estimate was used)

No extra Python dependencies beyond numpy + pandas (already required by the
pipeline).  Charts are rendered by Chart.js loaded from CDN; the rest of the
page is fully functional even without internet access.
"""

from __future__ import annotations

import base64
import datetime
import json
import math
import os
import re

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Data-collection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_bed_col(path: str, col: int) -> list[float]:
    """Return column *col* (0-indexed) of a BED file as a list of floats."""
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path, sep="\t", header=None, comment="#")
        if df.shape[1] <= col:
            return []
        return pd.to_numeric(df.iloc[:, col], errors="coerce").dropna().tolist()
    except Exception:
        return []


def _count_bed_lines(path: str) -> int:
    """Return the number of data rows in a BED file (ignores comment lines)."""
    if not os.path.exists(path):
        return 0
    try:
        df = pd.read_csv(path, sep="\t", header=None, comment="#")
        return len(df)
    except Exception:
        return 0


def _read_bed_widths(path: str) -> list[int]:
    """Return peak widths (end - start, bp) from BED columns 1 and 2 (0-indexed)."""
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path, sep="\t", header=None, comment="#", usecols=[1, 2])
        widths = (df.iloc[:, 1] - df.iloc[:, 0]).tolist()
        return [int(w) for w in widths if w > 0]
    except Exception:
        return []


def _histogram(values: list[float], n_bins: int = 40) -> tuple[list[str], list[int]]:
    """Return (mid-point labels, counts) for a linear-scale histogram."""
    if not values:
        return [], []
    counts, edges = np.histogram(values, bins=n_bins)
    labels = [f"{(edges[i] + edges[i + 1]) / 2:.0f}" for i in range(len(edges) - 1)]
    return labels, counts.tolist()


def _log10_histogram(
    genic: list[float],
    noov: list[float],
    n_bins: int = 50,
) -> tuple[list[str], list[int], list[int]]:
    """
    Return unified log10 histogram bins for genic and non-overlapping peaks.
    Both series share the same x-axis edges so they overlay cleanly.
    """
    all_pos = [v for v in genic + noov if v > 0]
    if not all_pos:
        return [], [], []
    log_all = np.log10(all_pos)
    _, edges = np.histogram(log_all, bins=n_bins)

    def _bin(vals: list[float]) -> list[int]:
        pos = [v for v in vals if v > 0]
        if not pos:
            return [0] * (len(edges) - 1)
        c, _ = np.histogram(np.log10(pos), bins=edges)
        return c.tolist()

    labels = [f"{(edges[i] + edges[i + 1]) / 2:.3f}" for i in range(len(edges) - 1)]
    return labels, _bin(genic), _bin(noov)


def _count_bam_reads(bam_path: str) -> int:
    """Count alignment records in a BAM file using samtools view -c."""
    import subprocess
    try:
        r = subprocess.run(
            ["samtools", "view", "-c", bam_path],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode == 0:
            return int(r.stdout.strip())
    except Exception:
        pass
    return 0


def _read_image_b64(path: str) -> str:
    """Read an image file as base64 string (empty when unavailable)."""
    try:
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode()
    except Exception:
        return ""


def _get_reads_info(tempdir: str) -> dict:
    """
    Return the number of reads used for peak calling and whether the BAM was subsampled.

    Priority:
      1. subsampled.bam  — used when --subsamplebam was set
      2. plus.bam + minus.bam — the strand-split BAMs actually fed to MACS2
      3. MACS2 xls "total tags" — fallback if BAM files are unavailable
    """
    subsampled_bam = os.path.join(tempdir, "subsampled.bam")
    subsampled = os.path.exists(subsampled_bam)

    # Option 1: subsampled.bam
    if subsampled:
        n = _count_bam_reads(subsampled_bam)
        if n > 0:
            return {"n_reads": n, "subsampled": True}

    # Option 2: plus.bam + minus.bam
    total = 0
    for fname in ("plus.bam", "minus.bam"):
        p = os.path.join(tempdir, fname)
        if os.path.exists(p):
            total += _count_bam_reads(p)
    if total > 0:
        return {"n_reads": total, "subsampled": subsampled}

    # Option 3: MACS2 xls fallback
    xls_total = 0
    for fname in ("plus_peaks.xls", "minus_peaks.xls"):
        path = os.path.join(tempdir, fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as fh:
                for line in fh:
                    if line.startswith("# total tags in treatment:"):
                        xls_total += int(line.split(":")[1].strip())
                        break
        except Exception:
            pass
    return {"n_reads": xls_total, "subsampled": subsampled}


def _parse_mapping_stats(path: str) -> list[dict]:
    """
    Parse mapping_stats.txt (written by run_estimate).
    Returns a list of dicts — one per annotation (before / after).
    """
    if not os.path.exists(path):
        return []
    results: list[dict] = []
    cur: dict = {}

    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            # Header line usually looks like: "/path/to/file.gtf:"
            # but allow basename-only labels as well.
            if line.endswith(":") and not re.match(
                r"^(Total reads|Mapped reads|Genic reads|Orphan peaks|Intergenic reads):",
                line,
            ):
                if cur:
                    results.append(cur)
                cur = {"label": line.rstrip(":")}
            elif line.startswith("Total reads"):
                cur["total"] = int(line.split(": ")[1].split()[0])
            elif line.startswith("Mapped reads"):
                parts = line.split("(total: ")
                cur["mapped"] = int(parts[0].split(": ")[1].strip())
                cur["mapped_pct"] = float(parts[1].split("%")[0].strip())
            elif line.startswith("Genic reads"):
                parts = line.split("(total: ")
                cur["genic"] = int(parts[0].split(": ")[1].strip())
                cur["genic_pct"] = float(parts[1].split("%")[0].strip())
            elif line.startswith("Orphan peaks"):
                parts = line.split("(total: ")
                cur["orphan"] = int(parts[0].split(": ")[1].strip())
                cur["orphan_pct"] = float(parts[1].split("%")[0].strip())
            elif line.startswith("Intergenic reads"):
                parts = line.split("(total: ")
                cur["intergenic"] = int(parts[0].split(": ")[1].strip())
                cur["intergenic_pct"] = float(parts[1].split("%")[0].strip())

    if cur:
        results.append(cur)
    return results


def _read_fix_info(tempdir: str) -> dict:
    """
    Read structured fix metadata emitted by geneext.py.
    """
    default = {
        "schema": "v1",
        "input_genefile": "",
        "final_genefile": "",
        "rerun_mode": False,
        "force_mode": False,
        "skipped_steps": [],
        "extension_param": {
            "name": "--maxdist",
            "mode": "",
            "user_value_bp": None,
            "effective_value_bp": None,
            "auto_quantile": None,
        },
        "steps": {
            "mRNA_to_transcript": {"applied": False},
            "gene_features_added": {"applied": False, "n_genes_added": 0, "gene_ids_file": ""},
            "clip_5prime": {"applied": False, "n_events": 0, "n_genes_clipped": 0, "log_file": "", "gene_ids_file": ""},
        },
    }
    path = os.path.join(tempdir, "report_fix_info.json")
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            return default
        steps = raw.get("steps", {}) if isinstance(raw.get("steps", {}), dict) else {}
        skipped_steps = raw.get("skipped_steps", [])
        if not isinstance(skipped_steps, list):
            skipped_steps = []
        ext_param = raw.get("extension_param", {})
        if not isinstance(ext_param, dict):
            ext_param = {}
        out = default.copy()
        out.update({
            k: raw.get(k, out.get(k))
            for k in ["schema", "input_genefile", "final_genefile", "rerun_mode", "force_mode"]
        })
        out["skipped_steps"] = [str(x) for x in skipped_steps]
        out["extension_param"] = {
            "name": ext_param.get("name", out["extension_param"]["name"]),
            "mode": ext_param.get("mode", out["extension_param"]["mode"]),
            "user_value_bp": ext_param.get("user_value_bp", out["extension_param"]["user_value_bp"]),
            "effective_value_bp": ext_param.get("effective_value_bp", out["extension_param"]["effective_value_bp"]),
            "auto_quantile": ext_param.get("auto_quantile", out["extension_param"]["auto_quantile"]),
        }
        out["steps"] = {
            "mRNA_to_transcript": steps.get("mRNA_to_transcript", out["steps"]["mRNA_to_transcript"]),
            "gene_features_added": steps.get("gene_features_added", out["steps"]["gene_features_added"]),
            "clip_5prime": steps.get("clip_5prime", out["steps"]["clip_5prime"]),
        }
        return out
    except Exception:
        return default


def _parse_run_log(path: str) -> dict:
    """
    Parse GeneExt textual log for command, stages and notable messages.
    """
    out = {
        "exists": False,
        "command": "",
        "sections": [],
        "notes": [],
        "genome_fix_detected": False,
        "log_file": os.path.basename(path),
    }
    if not os.path.exists(path):
        return out

    out["exists"] = True
    try:
        with open(path, "r", errors="replace") as fh:
            lines = [ln.rstrip("\n") for ln in fh]
    except Exception:
        return out

    cmd_pat = re.compile(r"^(python(?:\d+(?:\.\d+)?)?\s+.*\bgeneext\.py\b|.*\bgeneext\.py\b.*)$")
    for ln in lines:
        txt = ln.strip()
        if not txt:
            continue
        if cmd_pat.search(txt):
            out["command"] = txt
            break

    seen_sections = set()
    notes = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m = re.search(r"^│\s*(.*?)\s*│$", line)
        if m:
            sec = m.group(1).strip()
            if sec and sec not in seen_sections:
                seen_sections.add(sec)
                out["sections"].append(sec)

        low = line.lower()
        if any(k in low for k in ["trying to fix", "fixed genome", "found fixed genome", "--onlyfix", "5' clipping"]):
            out["genome_fix_detected"] = True

        if any(k in low for k in ["warning", "error", "trying to fix", "found fixed genome", "subsampling", "filtering", "peak", "extension"]):
            if line not in notes:
                notes.append(line)

    out["notes"] = notes[:8]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def generate_html_report(
    tempdir: str,
    outputfile: str,
    genefile: str,
    infmt: str,
    coverage_percentile: float,
    count_threshold=None,
    do_estimate: bool = False,
    n_genes: int = 0,
    run_args: str = "",
    orphan_warn_fraction: float = 0.10,
) -> str:
    """
    Build the web-summary HTML and write it next to the output annotation.

    Parameters
    ----------
    tempdir            : directory containing intermediate pipeline files
    outputfile         : final annotation path (report lands at outputfile + '.Report.html')
    genefile           : original input annotation path
    infmt              : 'gtf' | 'gff' | 'bed'
    coverage_percentile: percentile used for peak filtering  (0 = disabled)
    count_threshold    : actual coverage cutoff value (may be None)
    do_estimate        : whether mapping estimation was run
    n_genes            : total gene count in the original annotation
    run_args           : command-line string for display in the report
    orphan_warn_fraction: warning threshold for orphan peaks relative to input
                         genes (warn if orphan_peaks > fraction * n_genes)
    """
    # ── Extension lengths + table ─────────────────────────────────────────────
    ext_path = os.path.join(tempdir, "extensions.tsv")
    ext_lengths: list[float] = []
    ext_table: list[dict] = []
    if os.path.exists(ext_path):
        try:
            df_ext = pd.read_csv(ext_path, sep="\t", header=None)
            ext_lengths = (
                pd.to_numeric(df_ext.iloc[:, -1], errors="coerce").dropna().tolist()
            )
            if df_ext.shape[1] >= 3:
                for _, row in df_ext.iterrows():
                    ext_val = pd.to_numeric(row.iloc[2], errors="coerce")
                    ext_table.append({
                        "gene": str(row.iloc[0]),
                        "peak": str(row.iloc[1]),
                        "ext":  int(ext_val) if not pd.isna(ext_val) else 0,
                    })
        except Exception:
            pass

    # ── Logo (embedded as base64 for self-contained HTML) ────────────────────
    img_dir = os.path.join(os.path.dirname(__file__), "..", "img")
    logo_b64 = _read_image_b64(os.path.join(img_dir, "logo.png"))
    manual_figs = {
        "max_ext": _read_image_b64(os.path.join(img_dir, "max_ext.png")),
        "peak_filtering": _read_image_b64(os.path.join(img_dir, "peak_filtering.png")),
    }

    n_extended  = len(ext_lengths)
    pct_extended = round(n_extended / n_genes * 100, 1) if n_genes > 0 else 0.0
    min_ext      = round(float(np.min(ext_lengths)),    1) if ext_lengths else 0.0
    median_ext   = round(float(np.median(ext_lengths)), 1) if ext_lengths else 0.0
    mean_ext     = round(float(np.mean(ext_lengths)),   1) if ext_lengths else 0.0
    max_ext      = round(float(np.max(ext_lengths)),    1) if ext_lengths else 0.0

    ext_labels, ext_counts = _histogram(ext_lengths, n_bins=100)

    # ── Peak coverages ───────────────────────────────────────────────────────
    genic_cov = _read_bed_col(os.path.join(tempdir, "genic_peaks.bed"),   6)
    noov_cov  = _read_bed_col(os.path.join(tempdir, "allpeaks_noov.bed"), 6)

    cov_labels, cov_genic, cov_noov = _log10_histogram(genic_cov, noov_cov, n_bins=50)

    log10_thr: float | None = None
    if count_threshold is not None:
        try:
            v = float(count_threshold)
            if v > 0:
                log10_thr = round(math.log10(v), 4)
        except Exception:
            pass

    # ── Orphan peaks (prefer merged clusters) ────────────────────────────────
    orphan_bed_text = ""
    n_orphan_merged = 0
    for _fname in ("orphan_merged.bed", "orphan.bed"):
        _p = os.path.join(tempdir, _fname)
        if os.path.exists(_p):
            try:
                with open(_p) as _f:
                    orphan_bed_text = _f.read()
                n_orphan_merged = sum(
                    1
                    for line in orphan_bed_text.splitlines()
                    if line.strip() and not line.startswith("#")
                )
            except Exception:
                pass
            break

    # ── Peak-flow metrics for report diagram ─────────────────────────────────
    n_initial_called = _count_bed_lines(os.path.join(tempdir, "allpeaks.bed"))
    filtered_path = ""
    for _fname in ("allpeaks_noov_fcov.bed", "allpeaks_noov.bed"):
        _p = os.path.join(tempdir, _fname)
        if os.path.exists(_p):
            filtered_path = _p
            break
    n_passed_filter = _count_bed_lines(filtered_path) if filtered_path else 0
    n_assigned_gene = len(
        {
            str(r.get("peak", "")).strip()
            for r in ext_table
            if str(r.get("peak", "")).strip()
        }
    )
    peak_flow = {
        "has_macs2_peaks": n_initial_called > 0,
        "initial_called": n_initial_called,
        "passed_filtering": n_passed_filter,
        "assigned_to_genes": n_assigned_gene,
        "orphan_enabled": bool(orphan_bed_text),
        "orphan_count": n_orphan_merged,
        "filtered_file": os.path.basename(filtered_path) if filtered_path else "",
    }

    # ── Reads (from MACS2 xls) ───────────────────────────────────────────────
    reads_info = _get_reads_info(tempdir)

    # ── Mapping stats ────────────────────────────────────────────────────────
    mapping_stats = (
        _parse_mapping_stats(os.path.join(tempdir, "mapping_stats.txt"))
        if do_estimate
        else []
    )

    # ── Assemble payload ─────────────────────────────────────────────────────
    input_basename = os.path.basename(genefile)
    fix_info = _read_fix_info(tempdir)
    log_info = _parse_run_log(outputfile + ".GeneExt.log")
    gf_added = bool(fix_info.get("steps", {}).get("gene_features_added", {}).get("applied"))
    mrna_fixed = bool(fix_info.get("steps", {}).get("mRNA_to_transcript", {}).get("applied"))
    clipped5 = bool(fix_info.get("steps", {}).get("clip_5prime", {}).get("applied"))
    genome_fixed = (
        ".fixed." in input_basename
        or input_basename.startswith("genome.fixed.")
        or bool(log_info.get("genome_fix_detected"))
        or gf_added
        or mrna_fixed
        or clipped5
    )
    command_text = run_args or log_info.get("command")
    # Keep subsampling flag robust even when subsampled.bam is missing from tmp.
    # This can happen in rerun/cleanup scenarios, but the run still used subsampling.
    cmd_low = (command_text or "").lower()
    if ("--subsamplebam" in cmd_low) or ("-subsamplebam" in cmd_low):
        reads_info["subsampled"] = True

    orphan_gene_fraction = (n_orphan_merged / n_genes) if n_genes > 0 else None
    orphan_warning = bool(
        n_genes > 0 and n_orphan_merged > (float(orphan_warn_fraction) * float(n_genes))
    )

    payload = {
        "summary": {
            "n_genes":        n_genes,
            "n_extended":     n_extended,
            "pct_extended":   pct_extended,
            "min_ext":        min_ext,
            "median_ext":     median_ext,
            "mean_ext":       mean_ext,
            "max_ext":        max_ext,
            "n_genic_peaks":  len(genic_cov),
            "n_noov_peaks":   len(noov_cov),
            "n_orphan_peaks": n_orphan_merged,
            "orphan_warn_fraction": orphan_warn_fraction,
            "orphan_gene_fraction": orphan_gene_fraction,
            "orphan_warning": orphan_warning,
            "cov_percentile": coverage_percentile,
            "cov_threshold":  count_threshold,
            "n_reads":        reads_info["n_reads"],
            "subsampled":     reads_info["subsampled"],
            "output_file":    os.path.basename(outputfile),
            "input_file":     input_basename,
            "genome_fixed":   genome_fixed,
            "log_genome_fix": bool(log_info.get("genome_fix_detected")),
            "log_file":       log_info.get("log_file") if log_info.get("exists") else "",
            "run_date":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "run_args":       command_text,
        },
        "ext_hist":  {"labels": ext_labels, "counts": ext_counts},
        "cov_hist":  {
            "labels":         cov_labels,
            "counts_genic":   cov_genic,
            "counts_noov":    cov_noov,
            "log10_threshold": log10_thr,
        },
        "mapping_stats": mapping_stats,
        "ext_table": ext_table,
        "orphan_bed": orphan_bed_text,
        "peak_flow": peak_flow,
        "log_sections": log_info.get("sections", []),
        "log_notes": log_info.get("notes", []),
        "fix_info": fix_info,
    }

    html = _render_html(payload, logo_b64=logo_b64, manual_figs=manual_figs)
    report_path = outputfile + ".Report.html"
    with open(report_path, "w") as fh:
        fh.write(html)
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────

def _render_html(data: dict, logo_b64: str = "", manual_figs: dict | None = None) -> str:
    payload_json = _safe_json_for_html(data)
    s = data["summary"]
    manual_figs = manual_figs or {}
    max_ext_b64 = manual_figs.get("max_ext", "")
    peak_filter_b64 = manual_figs.get("peak_filtering", "")
    logo_html = (
        f'<img src="data:image/png;base64,{logo_b64}"'
        f' style="display:block">'
        if logo_b64 else
        '<div class="logo-icon">G</div>'
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>GeneExt — {s['output_file']}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{font-size:17px}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
      background:#f0f2f5;color:#1a2332;min-height:100vh}}

/* ── header ─────────────────────────────────────────────────────────────── */
header{{background:#ffffff;color:#1a2332;box-shadow:0 2px 12px rgba(0,0,0,.15)}}
.header-inner{{max-width:1200px;margin:0 auto;padding:14px 24px;min-height:138px;
              display:flex;align-items:center;gap:16px;position:relative}}
.logo{{display:flex;align-items:center;justify-content:flex-start;flex:1}}
.logo img{{height:120px;max-width:72vw;width:auto;display:block;object-fit:contain}}
.logo-icon{{width:64px;height:64px;background:linear-gradient(135deg,#00b4d8,#48cae4);
            border-radius:12px;display:flex;align-items:center;justify-content:center;
            font-size:26px;font-weight:900;color:#0f3460;flex-shrink:0}}
.run-meta{{margin-left:auto;text-align:right;font-size:.72rem;opacity:.85;line-height:1.8;color:#42566f;
           display:flex;flex-direction:column;justify-content:center;min-width:230px}}
@media(max-width:900px){{
  .header-inner{{padding:12px 24px;min-height:108px}}
  .logo img{{height:88px;max-width:68vw}}
  .run-meta{{font-size:.64rem;line-height:1.55;min-width:0}}
}}

/* ── main container ──────────────────────────────────────────────────────── */
main{{max-width:1200px;margin:28px auto;padding:0 24px}}

/* ── section title ───────────────────────────────────────────────────────── */
.section-title{{font-size:.7rem;font-weight:700;letter-spacing:1.2px;
                text-transform:uppercase;color:#5a7194;margin-bottom:12px;
                padding-bottom:6px;border-bottom:2px solid #dce3ee}}

/* ── stat cards ──────────────────────────────────────────────────────────── */
.card-group-label{{font-size:.65rem;font-weight:700;letter-spacing:1px;
                   text-transform:uppercase;color:#8695a8;margin:20px 0 8px;
                   padding-left:2px}}
.card-group-label:first-child{{margin-top:0}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
        gap:14px;margin-bottom:6px}}
.card{{background:#fff;border-radius:12px;padding:18px 20px;
       box-shadow:0 1px 4px rgba(0,0,0,.08);border-top:3px solid transparent;
       transition:box-shadow .15s}}
.card:hover{{box-shadow:0 4px 16px rgba(0,0,0,.12)}}
.card.accent-teal  {{border-top-color:#00b4d8}}
.card.accent-green {{border-top-color:#06d6a0}}
.card.accent-orange{{border-top-color:#ff9f43}}
.card.accent-purple{{border-top-color:#a55eea}}
.card.accent-blue  {{border-top-color:#4d9ef5}}
.card.accent-red   {{border-top-color:#ee5a6f}}
.card-value{{font-size:1.9rem;font-weight:800;line-height:1;color:#1a2332}}
.card-value span{{font-size:1rem;font-weight:600;color:#8695a8;margin-left:3px}}
.card-label{{font-size:.72rem;font-weight:600;color:#8695a8;
             text-transform:uppercase;letter-spacing:.6px;margin-top:6px}}

/* ── charts ──────────────────────────────────────────────────────────────── */
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px}}
@media(max-width:780px){{.charts{{grid-template-columns:1fr}}}}
.chart-card{{background:#fff;border-radius:12px;padding:20px 22px;
             box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.chart-card h3{{font-size:.85rem;font-weight:700;color:#1a2332;margin-bottom:4px}}
.chart-card p{{font-size:.72rem;color:#8695a8;margin-bottom:14px}}
.chart-wrap{{position:relative;height:260px}}
.manual-fig{{margin-top:10px;border:1px solid #dce3ee;border-radius:8px;overflow:hidden;background:#fff}}
.manual-fig img{{display:block;width:100%;height:auto}}
.manual-fig .cap{{padding:8px 10px;font-size:.68rem;color:#5a7194;background:#f9fbff;border-top:1px solid #eef1f7}}
.maxext-fig{{margin-top:12px}}

/* ── extension parameter section ────────────────────────────────────────── */
.ext-param-layout{{display:grid;grid-template-columns:minmax(260px,1fr) 2fr;gap:20px;margin-bottom:28px}}
@media(max-width:1150px){{.ext-param-layout{{grid-template-columns:1fr}}}}
.ext-param-card{{background:#fff;border-radius:12px;padding:18px 20px;
                 box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.ext-param-card h3{{font-size:.85rem;font-weight:700;color:#1a2332;margin-bottom:6px}}
.ext-param-card p{{font-size:.78rem;color:#5a7194;line-height:1.55;margin:4px 0}}

/* ── shared widget-card styling (matches summary cards look) ───────────── */
.widget-card{{border:1px solid #dce3ee;border-top:3px solid var(--accent,#00b4d8)}}
.widget-accent-teal{{--accent:#00b4d8}}
.widget-accent-green{{--accent:#06d6a0}}
.widget-accent-blue{{--accent:#4d9ef5}}
.widget-accent-purple{{--accent:#a55eea}}
.widget-accent-red{{--accent:#ee5a6f}}
.widget-accent-orange{{--accent:#ff9f43}}

/* ── peak-flow diagram ───────────────────────────────────────────────────── */
.flow-card{{background:#fff;border-radius:12px;padding:20px 22px;
           box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:28px}}
.peak-flow-layout{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px}}
@media(max-width:900px){{.peak-flow-layout{{grid-template-columns:1fr}}}}
.flow-card.compact{{margin-bottom:0}}
.flow-row{{display:flex;flex-direction:column;align-items:center;gap:8px}}
.flow-node{{background:#f7f9fd;border:1px solid #dce3ee;border-radius:10px;
            border-top:3px solid transparent;padding:12px 14px;min-width:220px}}
.flow-node .n{{font-size:1.9rem;font-weight:800;color:#1a2332;line-height:1}}
.flow-node .l{{font-size:.7rem;color:#5a7194;text-transform:uppercase;letter-spacing:.5px;margin-top:3px}}
.flow-node.accent-teal{{border-top-color:#00b4d8}}
.flow-node.accent-green{{border-top-color:#06d6a0}}
.flow-node.accent-blue{{border-top-color:#4d9ef5}}
.flow-node.accent-purple{{border-top-color:#a55eea}}
.flow-arrow{{color:#8fa1b8;font-size:1.1rem;font-weight:700;line-height:1}}
.flow-arrow.down{{margin-left:0}}
.flow-branch{{margin-top:8px;display:flex;flex-direction:column;align-items:center;gap:6px;color:#5a7194;font-size:.75rem;text-align:center}}
.flow-warn{{margin-bottom:12px;padding:10px 12px;border:1px solid #ffd7dc;border-top:3px solid #ee5a6f;
            border-radius:10px;background:#fff5f6;color:#7a2f39;font-size:.76rem;line-height:1.45}}

/* ── mapping table ───────────────────────────────────────────────────────── */
.table-card{{background:#fff;border-radius:12px;padding:20px 22px;
             box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:28px}}
.table-card h3{{font-size:.85rem;font-weight:700;color:#1a2332;margin-bottom:12px}}
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
thead tr{{background:#f4f7fb}}
th{{padding:9px 14px;text-align:left;font-weight:700;color:#5a7194;
    font-size:.72rem;text-transform:uppercase;letter-spacing:.5px}}
td{{padding:9px 14px;border-bottom:1px solid #eef1f7;color:#2d3e50}}
tr:last-child td{{border-bottom:none}}
.pct-bar{{display:flex;align-items:center;gap:8px}}
.bar-bg{{flex:1;height:7px;background:#eef1f7;border-radius:4px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:4px;background:#4d9ef5}}
.bar-fill.genic{{background:#06d6a0}}
.bar-fill.intergenic{{background:#ee5a6f}}
.bar-fill.orphan{{background:#ff9f43}}
td.num{{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}}
.diff-row{{background:#f9fbff;font-weight:700}}
.diff-row td{{color:#0f3460}}

/* ── args box ────────────────────────────────────────────────────────────── */
.args-box{{background:#fff;border-radius:12px;padding:16px 20px;
           box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:28px}}
.args-box pre{{font-size:.73rem;color:#5a7194;white-space:pre-wrap;
               word-break:break-all;line-height:1.55}}
.args-meta{{margin-top:10px;font-size:.72rem;color:#5a7194;line-height:1.5}}
.args-meta p{{margin:2px 0}}
.args-meta ul{{margin:8px 0 0 16px;padding:0}}
.args-meta li{{margin:2px 0}}

/* ── run status pane ─────────────────────────────────────────────────────── */
.status-box{{padding:14px 18px;font-size:.8rem;line-height:1.5;color:#2d3e50}}
.status-box strong{{font-size:.84rem}}
.status-box ul{{margin:8px 0 0 18px;padding:0}}
.status-box li{{margin:2px 0}}
.status-ok{{background:#f3fbf6;color:#2f6b46;border-color:#cfead8;border-top-color:#2f8f46}}
.status-warn{{background:#fff5f6;color:#7a2f39;border-color:#ffd7dc;border-top-color:#ee5a6f}}

/* ── footer ──────────────────────────────────────────────────────────────── */
footer{{text-align:center;font-size:.7rem;color:#aab4c0;padding:20px 0 32px}}

/* ── chart-unavailable banner ────────────────────────────────────────────── */
.chart-unavailable{{display:none;align-items:center;justify-content:center;
                    height:100%;font-size:.8rem;color:#aab4c0;flex-direction:column;gap:6px}}

/* ── clickable card ──────────────────────────────────────────────────────── */
.card.clickable{{cursor:pointer}}
.card.clickable:hover{{box-shadow:0 6px 20px rgba(0,0,0,.16);transform:translateY(-1px)}}

/* ── download buttons ────────────────────────────────────────────────────── */
.btn-dl{{display:inline-flex;align-items:center;gap:5px;padding:5px 12px;
         border-radius:7px;font-size:.72rem;font-weight:600;cursor:pointer;
         border:1px solid #dce3ee;background:#f4f7fb;color:#5a7194;
         text-decoration:none;transition:all .15s;white-space:nowrap}}
.btn-dl:hover{{background:#e8f4fd;border-color:#00b4d8;color:#0f3460}}
.btn-dl-sm{{display:inline-block;font-size:.62rem;color:#00b4d8;
            margin-top:4px;cursor:pointer}}
.btn-dl-sm:hover{{text-decoration:underline}}

/* ── modal ───────────────────────────────────────────────────────────────── */
.modal-backdrop{{display:none;position:fixed;inset:0;background:rgba(10,22,40,.55);
                 z-index:1000;align-items:center;justify-content:center}}
.modal-backdrop.open{{display:flex}}
.modal{{background:#fff;border-radius:14px;width:min(860px,94vw);max-height:82vh;
        display:flex;flex-direction:column;box-shadow:0 12px 48px rgba(0,0,0,.28)}}
.modal-header{{display:flex;align-items:center;justify-content:space-between;
               padding:18px 24px;border-bottom:1px solid #dce3ee}}
.modal-header h2{{font-size:.95rem;font-weight:700;color:#1a2332}}
.modal-close{{border:none;background:none;font-size:1.3rem;color:#8695a8;
              cursor:pointer;line-height:1;padding:2px 6px;border-radius:6px}}
.modal-close:hover{{background:#f0f2f5;color:#1a2332}}
.modal-search{{padding:12px 24px;border-bottom:1px solid #eef1f7}}
.modal-search input{{width:100%;padding:8px 12px;border:1px solid #dce3ee;
                     border-radius:8px;font-size:.82rem;outline:none}}
.modal-search input:focus{{border-color:#00b4d8}}
.modal-body{{overflow-y:auto;padding:0}}
.modal-body table{{width:100%;border-collapse:collapse;font-size:.82rem}}
.modal-body thead tr{{background:#f4f7fb;position:sticky;top:0}}
.modal-body th{{padding:9px 16px;text-align:left;font-weight:700;color:#5a7194;
                font-size:.72rem;text-transform:uppercase;letter-spacing:.5px;
                cursor:pointer;user-select:none;white-space:nowrap}}
.modal-body th:hover{{color:#0f3460}}
.modal-body th .sort-icon{{margin-left:4px;opacity:.4}}
.modal-body th.sort-active .sort-icon{{opacity:1}}
.modal-body td{{padding:8px 16px;border-bottom:1px solid #eef1f7;color:#2d3e50}}
.modal-body tr:last-child td{{border-bottom:none}}
.modal-body tr:hover td{{background:#fafbff}}
.modal-footer{{padding:10px 24px;border-top:1px solid #eef1f7;font-size:.72rem;color:#8695a8}}
</style>
</head>
<body>

<header>
  <div class="header-inner">
    <div class="logo">
      {logo_html}
    </div>
    <div class="run-meta">
      <div>Output: <strong>{s['output_file']}</strong></div>
      <div>Input: {s['input_file']}</div>
      <div>{s['run_date']}</div>
    </div>
  </div>
</header>

<main>

  <!-- ── Run status ─────────────────────────────────────────────────────── -->
  <div id="reportStatusSection"></div>

  <!-- ── Command args ──────────────────────────────────────────────────── -->
  <div id="argsSection"></div>
  <div id="executionSummarySection"></div>
  <div id="genomeFixSection"></div>

  <!-- ── Summary cards ─────────────────────────────────────────────────── -->
  <p class="section-title">Run summary</p>
  <div id="summaryCards"></div>

  <!-- ── Extension parameter + extension distribution ──────────────────── -->
  <p class="section-title">Extension Parameter</p>
  <div class="ext-param-layout">
    <div class="ext-param-card widget-card widget-accent-orange">
      <h3>Maximal extension length (<code>-m</code> / <code>--maxdist</code>)</h3>
      <div id="extensionParamSummary" class="args-meta"></div>
      {(
        '<div class="manual-fig maxext-fig"><img alt="Maximal extension length explainer" src="data:image/png;base64,'
        + max_ext_b64
        + '"><div class="cap">Maximal extension length (<code>-m</code>) shown with the parameter context on the left.</div></div>'
      ) if max_ext_b64 else ''}
    </div>
    <div class="chart-card widget-card widget-accent-blue">
      <h3>3&prime;-extension length distribution</h3>
      <p>Number of base-pairs each gene model was extended at its 3&prime; end</p>
      <p id="extStatsLabel" style="font-size:.72rem;color:#5a7194;margin-bottom:10px"></p>
      <div class="chart-wrap">
        <canvas id="extChart"></canvas>
        <div class="chart-unavailable" id="extNA">
          <span>&#9888;</span>No extension data available
        </div>
      </div>
    </div>
  </div>

  <!-- ── Peak filtering flow + coverage distribution ───────────────────── -->
  <p class="section-title">Peak Filtering Flow</p>
  <div class="peak-flow-layout">
    <div id="peakFlowSection"></div>
    <div class="chart-card">
      <h3>Peak coverage filtering (log&#x2081;&#x2080;)</h3>
      <p>
        <span style="color:#ff6384">&#9632;</span> Genic peaks (reference) &nbsp;
        <span style="color:#4d9ef5">&#9632;</span> Intergenic peaks (candidates)
        <span id="thrLabel"></span>
      </p>
      <div class="chart-wrap">
        <canvas id="covChart"></canvas>
        <div class="chart-unavailable" id="covNA">
          <span>&#9888;</span>No coverage data available
        </div>
      </div>
      {(
        '<div class="manual-fig"><img alt="Peak filtering explainer" src="data:image/png;base64,'
        + peak_filter_b64
        + '"><div class="cap">Intergenic peak filtering using <code>--peak_perc</code>.</div></div>'
      ) if peak_filter_b64 else ''}
    </div>
  </div>

  <!-- ── Mapping table (only if --estimate) ────────────────────────────── -->
  <div id="mappingSection"></div>

  <!-- ── Citation ───────────────────────────────────────────────────────── -->
  <p class="section-title">Citation</p>
  <div class="args-box widget-card widget-accent-purple">
    <p style="font-size:.84rem;line-height:1.65;color:#2d3e50;margin:0">
      Grygoriy Zolotarov, Xavier Grau-Bov&eacute;, Arnau Seb&eacute;-Pedr&oacute;s, GeneExt: a gene model extension tool for enhanced single-cell RNA-seq analysis, <em>Bioinformatics</em>, Volume 42, Issue 3, March 2026, btag094,
      <a href="https://doi.org/10.1093/bioinformatics/btag094" target="_blank" rel="noopener noreferrer">https://doi.org/10.1093/bioinformatics/btag094</a>
    </p>
  </div>

</main>

<!-- ── Extended genes modal ──────────────────────────────────────────────── -->
<div class="modal-backdrop" id="extModal" role="dialog" aria-modal="true">
  <div class="modal">
    <div class="modal-header">
      <h2>Extended genes</h2>
      <div style="display:flex;align-items:center;gap:10px">
        <button class="btn-dl" id="extDownloadBtn">&#8595; Download TSV</button>
        <button class="modal-close" id="extModalClose" aria-label="Close">&times;</button>
      </div>
    </div>
    <div class="modal-search">
      <input type="search" id="extSearch" placeholder="Filter by gene or peak ID&hellip;">
    </div>
    <div class="modal-body">
      <table id="extTable">
        <thead>
          <tr>
            <th data-col="gene">Gene ID <span class="sort-icon">&#9661;</span></th>
            <th data-col="peak">Peak ID <span class="sort-icon">&#9661;</span></th>
            <th data-col="ext" class="sort-active">Extension (bp) <span class="sort-icon">&#9660;</span></th>
          </tr>
        </thead>
        <tbody id="extTableBody"></tbody>
      </table>
    </div>
    <div class="modal-footer" id="extTableFooter"></div>
  </div>
</div>

<!-- ── Orphan peaks modal ──────────────────────────────────────────────── -->
<div class="modal-backdrop" id="orphanModal" role="dialog" aria-modal="true">
  <div class="modal">
    <div class="modal-header">
      <h2>Orphan peak clusters</h2>
      <div style="display:flex;align-items:center;gap:10px">
        <button class="btn-dl" id="orphanDownloadBtn">&#8595; Download BED</button>
        <button class="modal-close" id="orphanModalClose" aria-label="Close">&times;</button>
      </div>
    </div>
    <div class="modal-search">
      <input type="search" id="orphanSearch" placeholder="Filter by chromosome, peak ID, or strand&hellip;">
    </div>
    <div class="modal-body">
      <table id="orphanTable">
        <thead>
          <tr>
            <th data-col="chrom">Chrom <span class="sort-icon">&#9661;</span></th>
            <th data-col="start" class="sort-active">Start <span class="sort-icon">&#9650;</span></th>
            <th data-col="end">End <span class="sort-icon">&#9661;</span></th>
            <th data-col="id">Peak ID <span class="sort-icon">&#9661;</span></th>
            <th data-col="strand">Strand <span class="sort-icon">&#9661;</span></th>
          </tr>
        </thead>
        <tbody id="orphanTableBody"></tbody>
      </table>
    </div>
    <div class="modal-footer" id="orphanTableFooter"></div>
  </div>
</div>

<footer>Generated by GeneExt &bull; {s['run_date']}</footer>

<!-- Chart.js via CDN (requires internet); charts degrade gracefully offline -->
<script
  src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"
  crossorigin="anonymous"
  onerror="window.CHARTJS_FAILED=true"
></script>

<script>
const D = {payload_json};
const S = D.summary;

function escHtml(value) {{
  var safe = (value === null || value === undefined) ? '' : String(value);
  return safe
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}}

// ── Stat cards (grouped) ────────────────────────────────────────────────────
const readsLabel = S.n_reads
  ? (S.n_reads.toLocaleString() + (S.subsampled ? ' &#9432;' : ''))
  : '—';
const readsHint = S.subsampled
  ? '<div style="font-size:.62rem;color:#ff9f43;margin-top:4px">&#9888; BAM was subsampled</div>'
  : '';
const orphanWarnHint = S.orphan_warning
  ? '<div style="font-size:.62rem;color:#ee5a6f;margin-top:4px">&#9888; Too many orphan peaks/clusters</div>'
  : '';

function fmtInt(value) {{
  return (typeof value === 'number') ? value.toLocaleString() : value;
}}

// ── Run status pane ─────────────────────────────────────────────────────────
(function buildRunStatus() {{
  const warnings = [];
  if (S.orphan_warning) {{
    const orphanPct = Math.round(Number(S.orphan_gene_fraction || 0) * 100);
    const thrPct = Math.round(Number(S.orphan_warn_fraction || 0) * 100);
    warnings.push(
      `Too many orphan peaks: ${{fmtInt(S.n_orphan_peaks)}} orphan peaks/clusters ` +
      `vs ${{fmtInt(S.n_genes)}} input genes (${{orphanPct}}% > ${{thrPct}}%).`
    );
  }}

  if (warnings.length) {{
    const items = warnings.map(w => `<li>${{escHtml(w)}}</li>`).join('');
    document.getElementById('reportStatusSection').innerHTML = `
      <p class="section-title">Run Status</p>
      <div class="args-box widget-card status-box status-warn">
        <strong>&#9888; Warnings detected</strong>
        <ul>${{items}}</ul>
      </div>`;
  }} else {{
    document.getElementById('reportStatusSection').innerHTML = `
      <p class="section-title">Run Status</p>
      <div class="args-box widget-card status-box status-ok">
        <strong>&#10003; No warnings detected</strong>
        <p style="margin-top:6px">No report-level warnings were detected.</p>
      </div>`;
  }}
}})();

const CARD_GROUPS = [
  {{
    title: 'Gene Extension',
    cards: [
      {{value: fmtInt(S.n_extended), suffix: '/' + fmtInt(S.n_genes), label: 'Genes extended',     cls: 'accent-teal',   id: 'cardGenesExtended', clickable: true}},
      {{value: S.pct_extended, suffix: '%',            label: 'Extension rate',     cls: 'accent-green'}},
    ]
  }},
  {{
    title: 'Peaks',
    cards: [
      {{value: fmtInt(S.n_genic_peaks), suffix: '',  label: 'Genic peaks',              cls: 'accent-teal'}},
      {{value: fmtInt(S.n_noov_peaks),  suffix: '',  label: 'Intergenic candidate peaks',    cls: 'accent-blue'}},
      ...(S.n_orphan_peaks ? [{{value: fmtInt(S.n_orphan_peaks), suffix: '', label: 'Orphan peaks/clusters', cls: 'accent-purple', id: 'cardOrphanPeaks', clickable: true,
        extraHint: (D.orphan_bed ? '<span class="btn-dl-sm" onclick="downloadOrphanBed(event)">&#8595; Download BED</span>' : '') + orphanWarnHint}}] : []),
      {{value: S.cov_percentile || '—', suffix: S.cov_percentile ? 'th pct' : '', label: 'Intergenic filter (<code>--peak_perc</code>)', cls: 'accent-red'}},
    ]
  }},
  {{
    title: 'Reads',
    cards: [
      {{value: readsLabel, suffix: '', label: 'Reads used for MACS2 peak calling', cls: 'accent-teal', extraHint: readsHint}},
    ]
  }},
];

(function buildCards() {{
  const wrap = document.getElementById('summaryCards');
  CARD_GROUPS.forEach(g => {{
    if (!g.cards || !g.cards.length) return;
    wrap.innerHTML += `<p class="card-group-label">${{g.title}}</p>`;
    let row = '<div class="cards">';
    g.cards.forEach(c => {{
      const extra = c.clickable ? ' clickable' : '';
      const idAttr = c.id ? ` id="${{c.id}}"` : '';
      const hint = c.clickable
        ? '<div style="font-size:.62rem;color:#00b4d8;margin-top:4px">Click to view &#8599;</div>'
        : (c.extraHint || '');
      row += `<div class="card ${{c.cls}}${{extra}}"${{idAttr}}>
        <div class="card-value">${{c.value}}<span>${{c.suffix}}</span></div>
        <div class="card-label">${{c.label}}</div>
        ${{hint}}
      </div>`;
    }});
    row += '</div>';
    wrap.innerHTML += row;
  }});
  const geCard = document.getElementById('cardGenesExtended');
  if (geCard) geCard.addEventListener('click', openExtModal);
  const orphanCard = document.getElementById('cardOrphanPeaks');
  if (orphanCard) orphanCard.addEventListener('click', openOrphanModal);
}})();

// ── Peak filtering flow ───────────────────────────────────────────────────
(function buildPeakFlow() {{
  const PF = D.peak_flow || {{}};
  const orphanWarn = !!S.orphan_warning;
  const orphanFrac = Number(S.orphan_gene_fraction || 0);
  const orphanThr = Number(S.orphan_warn_fraction || 0);
  if (!PF.has_macs2_peaks) {{
    document.getElementById('peakFlowSection').innerHTML = `
      <div class="flow-card compact">
        <p style="font-size:.8rem;color:#8695a8">No MACS2 peak-calling output found; flow summary is unavailable.</p>
      </div>`;
    return;
  }}

  const fmt = v => Number(v || 0).toLocaleString();
  const orphanNode = PF.orphan_enabled
    ? `<div class="flow-branch">
         <span><code>--orphan</code> enabled (retained from intergenic filtered peaks)</span>
         <span class="flow-arrow down">&#8595;</span>
         <div class="flow-node accent-purple">
           <div class="n">${{fmt(PF.orphan_count)}}</div>
           <div class="l">Orphan peaks/clusters</div>
         </div>
       </div>`
    : '';
  const filteredNote = PF.filtered_file
    ? `<p style="font-size:.7rem;color:#8695a8;margin-top:8px">Filtered peaks source: ${{escHtml(PF.filtered_file)}}</p>`
    : '';
  const warningNote = orphanWarn
    ? `<div class="flow-warn">&#9888; <strong>Too many orphan peaks.</strong>
         ${{fmt(PF.orphan_count)}} orphan peaks vs ${{S.n_genes.toLocaleString()}} input genes
         (${{Math.round(orphanFrac * 100)}}% > ${{Math.round(orphanThr * 100)}}%).</div>`
    : '';

  document.getElementById('peakFlowSection').innerHTML = `
    <div class="flow-card compact">
      ${{warningNote}}
      <div class="flow-row">
        <div class="flow-node accent-teal">
          <div class="n">${{fmt(PF.initial_called)}}</div>
          <div class="l">Initial MACS2 peaks</div>
        </div>
        <span class="flow-arrow down">&#8595;</span>
        <div class="flow-node accent-green">
          <div class="n">${{fmt(PF.passed_filtering)}}</div>
          <div class="l">Intergenic peaks retained (<code>--peak_perc</code>)</div>
        </div>
        <span class="flow-arrow down">&#8595;</span>
        <div class="flow-node accent-blue">
          <div class="n">${{fmt(PF.assigned_to_genes)}}</div>
          <div class="l">Assigned to upstream genes</div>
        </div>
      </div>
      ${{orphanNode}}
      ${{filteredNote}}
    </div>`;
}})();

// ── Extended genes modal ────────────────────────────────────────────────────
let extSortCol = 'ext', extSortAsc = false;
let extFilteredRows = D.ext_table ? [...D.ext_table] : [];

function renderExtTable() {{
  const q = (document.getElementById('extSearch').value || '').toLowerCase();
  extFilteredRows = (D.ext_table || []).filter(r =>
    r.gene.toLowerCase().includes(q) || r.peak.toLowerCase().includes(q)
  );
  extFilteredRows.sort((a, b) => {{
    const va = a[extSortCol], vb = b[extSortCol];
    if (typeof va === 'number') return extSortAsc ? va - vb : vb - va;
    return extSortAsc ? String(va).localeCompare(vb) : String(vb).localeCompare(va);
  }});
  const tbody = document.getElementById('extTableBody');
  tbody.innerHTML = extFilteredRows.map(r => `
    <tr>
      <td>${{escHtml(r.gene)}}</td>
      <td>${{escHtml(r.peak)}}</td>
      <td style="text-align:right;font-variant-numeric:tabular-nums;font-weight:600">${{r.ext.toLocaleString()}}</td>
    </tr>`).join('');
  document.getElementById('extTableFooter').textContent =
    extFilteredRows.length + ' of ' + (D.ext_table || []).length + ' genes';
  // update sort icons
  document.querySelectorAll('#extTable th').forEach(th => {{
    const icon = th.querySelector('.sort-icon');
    if (th.dataset.col === extSortCol) {{
      th.classList.add('sort-active');
      icon.innerHTML = extSortAsc ? '&#9650;' : '&#9660;';
    }} else {{
      th.classList.remove('sort-active');
      icon.innerHTML = '&#9661;';
    }}
  }});
}}

function openExtModal() {{
  document.getElementById('extModal').classList.add('open');
  document.getElementById('extSearch').value = '';
  extSortCol = 'ext'; extSortAsc = false;
  renderExtTable();
}}

function closeExtModal() {{
  document.getElementById('extModal').classList.remove('open');
}}

document.getElementById('extModalClose').addEventListener('click', closeExtModal);
document.getElementById('extModal').addEventListener('click', e => {{
  if (e.target === document.getElementById('extModal')) closeExtModal();
}});
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') {{ closeExtModal(); closeOrphanModal(); }}
}});
document.getElementById('extSearch').addEventListener('input', renderExtTable);
document.querySelectorAll('#extTable th').forEach(th => {{
  th.addEventListener('click', () => {{
    if (extSortCol === th.dataset.col) extSortAsc = !extSortAsc;
    else {{ extSortCol = th.dataset.col; extSortAsc = th.dataset.col !== 'ext'; }}
    renderExtTable();
  }});
}});

// ── Orphan peaks modal ───────────────────────────────────────────────────
let orphanSortCol = 'start', orphanSortAsc = true;
let orphanRows = [];
let orphanFilteredRows = [];

function parseOrphanBed() {{
  if (!D.orphan_bed) return [];
  const rows = [];
  D.orphan_bed.split(/\\r?\\n/).forEach(line => {{
    if (!line || !line.trim() || line.startsWith('#')) return;
    const cols = line.split('\\t');
    rows.push({{
      chrom: cols[0] || '',
      start: Number(cols[1] || 0),
      end: Number(cols[2] || 0),
      id: cols[3] || '',
      strand: cols[5] || '',
    }});
  }});
  return rows;
}}

function renderOrphanTable() {{
  const q = (document.getElementById('orphanSearch').value || '').toLowerCase();
  orphanFilteredRows = orphanRows.filter(r =>
    String(r.chrom).toLowerCase().includes(q) ||
    String(r.id).toLowerCase().includes(q) ||
    String(r.strand).toLowerCase().includes(q)
  );

  orphanFilteredRows.sort((a, b) => {{
    const va = a[orphanSortCol], vb = b[orphanSortCol];
    if (typeof va === 'number') return orphanSortAsc ? va - vb : vb - va;
    return orphanSortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
  }});

  const tbody = document.getElementById('orphanTableBody');
  tbody.innerHTML = orphanFilteredRows.map(r => `
    <tr>
      <td>${{escHtml(r.chrom)}}</td>
      <td style="text-align:right;font-variant-numeric:tabular-nums">${{r.start.toLocaleString()}}</td>
      <td style="text-align:right;font-variant-numeric:tabular-nums">${{r.end.toLocaleString()}}</td>
      <td>${{escHtml(r.id)}}</td>
      <td>${{escHtml(r.strand)}}</td>
    </tr>`).join('');

  document.getElementById('orphanTableFooter').textContent =
    orphanFilteredRows.length + ' of ' + orphanRows.length + ' orphan peaks';

  document.querySelectorAll('#orphanTable th').forEach(th => {{
    const icon = th.querySelector('.sort-icon');
    if (th.dataset.col === orphanSortCol) {{
      th.classList.add('sort-active');
      icon.innerHTML = orphanSortAsc ? '&#9650;' : '&#9660;';
    }} else {{
      th.classList.remove('sort-active');
      icon.innerHTML = '&#9661;';
    }}
  }});
}}

function openOrphanModal() {{
  if (!orphanRows.length) orphanRows = parseOrphanBed();
  document.getElementById('orphanModal').classList.add('open');
  document.getElementById('orphanSearch').value = '';
  orphanSortCol = 'start'; orphanSortAsc = true;
  renderOrphanTable();
}}

function closeOrphanModal() {{
  document.getElementById('orphanModal').classList.remove('open');
}}

document.getElementById('orphanModalClose').addEventListener('click', closeOrphanModal);
document.getElementById('orphanModal').addEventListener('click', e => {{
  if (e.target === document.getElementById('orphanModal')) closeOrphanModal();
}});
document.getElementById('orphanSearch').addEventListener('input', renderOrphanTable);
document.querySelectorAll('#orphanTable th').forEach(th => {{
  th.addEventListener('click', () => {{
    if (orphanSortCol === th.dataset.col) orphanSortAsc = !orphanSortAsc;
    else {{ orphanSortCol = th.dataset.col; orphanSortAsc = ['start','end'].includes(th.dataset.col); }}
    renderOrphanTable();
  }});
}});

// ── Downloads ──────────────────────────────────────────────────────────────
function _download(text, filename, mime) {{
  const blob = new Blob([text], {{type: mime || 'text/plain'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}}

function downloadExtTsv() {{
  const rows = D.ext_table || [];
  const header = 'gene_id\\tpeak_id\\textension_bp\\n';
  const body = rows.map(r => r.gene + '\\t' + r.peak + '\\t' + r.ext).join('\\n');
  _download(header + body, S.output_file + '.extensions.tsv', 'text/tab-separated-values');
}}

function downloadOrphanBed(e) {{
  if (e) e.stopPropagation();
  if (!D.orphan_bed) return;
  _download(D.orphan_bed, 'orphan_peaks.bed', 'text/plain');
}}

document.getElementById('extDownloadBtn').addEventListener('click', downloadExtTsv);
document.getElementById('orphanDownloadBtn').addEventListener('click', downloadOrphanBed);

// ── Charts ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {{
  const extStats = document.getElementById('extStatsLabel');
  if (extStats) {{
    const hasExt = Number(S.n_extended || 0) > 0;
    extStats.innerHTML = hasExt
      ? `Min: <strong>${{S.min_ext}}</strong> bp &nbsp; Median: <strong>${{S.median_ext}}</strong> bp &nbsp; Max: <strong>${{S.max_ext}}</strong> bp`
      : 'No extension statistics available';
  }}

  if (window.CHARTJS_FAILED || typeof Chart === 'undefined') {{
    document.getElementById('extNA').style.display = 'flex';
    document.getElementById('covNA').style.display = 'flex';
    return;
  }}

  Chart.defaults.font.family =
    "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif";
  Chart.defaults.font.size = 11;
  Chart.defaults.color = '#5a7194';

  const EH = D.ext_hist;
  if (EH.counts && EH.counts.length) {{
    new Chart(document.getElementById('extChart'), {{
      type: 'bar',
      data: {{
        labels: EH.labels,
        datasets: [{{
          label: 'Genes',
          data: EH.counts,
          backgroundColor: 'rgba(0,180,216,0.65)',
          borderColor:     'rgba(0,180,216,0.9)',
          borderWidth: 1,
          borderRadius: 2,
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{display: false}},
          tooltip: {{
            callbacks: {{
              title: ctx => `~${{ctx[0].label}} bp`,
              label: ctx => ` ${{ctx.parsed.y}} gene(s)`
            }}
          }}
        }},
        scales: {{
          x: {{
            title: {{display:true, text:'Extension length (bp)', padding:{{top:6}}}},
            ticks: {{
              maxTicksLimit: 8,
              maxRotation: 0,
              callback: function(value) {{
                const raw = Number(this.getLabelForValue(value));
                if (!isFinite(raw)) return '';
                const base = raw >= 10000 ? 1000 : 100;
                const rounded = Math.round(raw / base) * base;
                return rounded.toLocaleString();
              }}
            }},
            grid: {{display: false}}
          }},
          y: {{
            title: {{display:true, text:'Number of genes'}},
            beginAtZero: true,
            grid: {{color:'#eef1f7'}}
          }}
        }}
      }}
    }});
  }} else {{
    document.getElementById('extNA').style.display = 'flex';
  }}

  const CH = D.cov_hist;
  if (CH.labels && CH.labels.length) {{
    // vertical annotation line for coverage threshold
    const thrPlugin = {{
      id: 'thrLine',
      afterDraw(chart) {{
        const thr = CH.log10_threshold;
        if (thr == null) return;
        const xAxis = chart.scales.x;
        const yAxis = chart.scales.y;
        const labels = chart.data.labels;
        // find nearest label index
        let nearest = 0, minD = Infinity;
        labels.forEach((l, i) => {{
          const d = Math.abs(parseFloat(l) - thr);
          if (d < minD) {{ minD = d; nearest = i; }}
        }});
        const x = xAxis.getPixelForValue(nearest);
        const ctx2 = chart.ctx;
        ctx2.save();
        ctx2.setLineDash([5, 4]);
        ctx2.strokeStyle = '#ee5a6f';
        ctx2.lineWidth = 1.5;
        ctx2.beginPath();
        ctx2.moveTo(x, yAxis.top);
        ctx2.lineTo(x, yAxis.bottom);
        ctx2.stroke();
        ctx2.restore();
      }}
    }};

    if (CH.log10_threshold != null) {{
      document.getElementById('thrLabel').innerHTML =
        ` &nbsp;<span style="color:#ee5a6f">&#9135;</span> threshold`;
    }}

    new Chart(document.getElementById('covChart'), {{
      type: 'bar',
      plugins: [thrPlugin],
      data: {{
        labels: CH.labels,
        datasets: [
          {{
            label: 'Genic peaks',
            data: CH.counts_genic,
            backgroundColor: 'rgba(255,99,132,0.5)',
            borderColor:     'rgba(255,99,132,0.8)',
            borderWidth: 1,
            borderRadius: 2,
          }},
          {{
            label: 'Non-overlapping peaks',
            data: CH.counts_noov,
            backgroundColor: 'rgba(77,158,245,0.45)',
            borderColor:     'rgba(77,158,245,0.8)',
            borderWidth: 1,
            borderRadius: 2,
          }}
        ]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{display: false}},
          tooltip: {{
            callbacks: {{
              title: ctx => `log10(cov) ≈ ${{ctx[0].label}}`
            }}
          }}
        }},
        scales: {{
          x: {{
            title: {{display:true, text:'log₁₀(normalized coverage)', padding:{{top:6}}}},
            ticks: {{maxTicksLimit: 8, maxRotation: 0}},
            grid: {{display: false}}
          }},
          y: {{
            title: {{display:true, text:'Number of peaks'}},
            beginAtZero: true,
            grid: {{color:'#eef1f7'}}
          }}
        }}
      }}
    }});
  }} else {{
    document.getElementById('covNA').style.display = 'flex';
  }}
}});

// ── Mapping stats table ────────────────────────────────────────────────────
(function buildMappingTable() {{
  const rows = D.mapping_stats;
  if (!rows || rows.length === 0) return;

  const cols = ['Total reads','Mapped','Genic','Orphan peaks','Intergenic'];
  const keys = ['total','mapped','genic','orphan','intergenic'];
  const pcts  = [null,'mapped_pct','genic_pct','orphan_pct','intergenic_pct'];
  const cls   = [null,'',          'genic',    'orphan',     'intergenic'];

  let html = `<div class="table-card">
    <h3>Mapping statistics</h3>
    <table>
      <thead><tr><th>Annotation</th>${{cols.map(c=>`<th>${{c}}</th>`).join('')}}</tr></thead>
      <tbody>`;

  rows.forEach(r => {{
    html += `<tr><td>${{escHtml(r.label.split('/').pop())}}</td>`;
    keys.forEach((k, i) => {{
      const v   = (r[k] === null || r[k] === undefined) ? '—' : r[k];
      const pct = pcts[i] ? ((r[pcts[i]] === null || r[pcts[i]] === undefined) ? null : r[pcts[i]]) : null;
      if (pct !== null) {{
        html += `<td><div class="pct-bar">
          <span>${{v.toLocaleString()}}</span>
          <div class="bar-bg"><div class="bar-fill ${{cls[i]}}" style="width:${{Math.min(pct,100)}}%"></div></div>
          <span style="min-width:42px;text-align:right">${{pct}}%</span>
        </div></td>`;
      }} else {{
        html += `<td class="num">${{typeof v==='number'?v.toLocaleString():v}}</td>`;
      }}
    }});
    html += `</tr>`;
  }});

  // diff row
  if (rows.length === 2 && rows[0].intergenic_pct != null && rows[1].intergenic_pct != null) {{
    const diff = Math.abs(rows[0].intergenic_pct - rows[1].intergenic_pct).toFixed(2);
    html += `<tr class="diff-row">
      <td colspan="${{cols.length + 1}}">
        &#x25B2; Intergenic read proportion reduced by ${{diff}}%
        (from ${{rows[0].intergenic_pct}}% → ${{rows[1].intergenic_pct}}%)
      </td></tr>`;
  }}

  html += `</tbody></table></div>`;
  document.getElementById('mappingSection').innerHTML = html;
}})();

// ── Extension parameter (--maxdist) ───────────────────────────────────────
(function buildExtensionParamSection() {{
  const F = D.fix_info || {{}};
  const E = F.extension_param || {{}};
  const mode = String(E.mode || '').toLowerCase();
  const eff = E.effective_value_bp;
  const user = E.user_value_bp;
  const q = E.auto_quantile;

  const hasData =
    (mode === 'user' || mode === 'auto') ||
    (eff !== null && eff !== undefined) ||
    (user !== null && user !== undefined);
  if (!hasData) {{
    const empty = document.getElementById('extensionParamSummary');
    if (empty) empty.innerHTML = '<p>No extension parameter metadata available.</p>';
    return;
  }}

  let rows = '';
  rows += `<p><strong>Parameter:</strong> <code>-m</code> / <code>--maxdist</code></p>`;
  if (mode === 'user') {{
    rows += `<p><strong>Selection mode:</strong> User-defined</p>`;
    if (user !== null && user !== undefined) {{
      rows += `<p><strong>User-defined maximal extension length:</strong> ${{Number(user).toLocaleString()}} bp</p>`;
    }}
  }} else if (mode === 'auto') {{
    rows += `<p><strong>Selection mode:</strong> Auto-estimated</p>`;
    if (q !== null && q !== undefined) {{
      rows += `<p><strong>Rule:</strong> ${{Math.round(Number(q) * 100)}}th percentile of gene genomic span (gene length quantile for <code>--maxdist</code>)</p>`;
    }}
  }}

  if (eff !== null && eff !== undefined) {{
    rows += `<p><strong>Final maximal extension length:</strong> ${{Number(eff).toLocaleString()}} bp</p>`;
  }}

  document.getElementById('extensionParamSummary').innerHTML = rows;
}})();

// ── Genome-fix summary box ─────────────────────────────────────────────────
(function buildGenomeFixSection() {{
  const F = D.fix_info || {{}};
  const steps = F.steps || {{}};
  const gf = steps.gene_features_added || {{}};
  const mrna = steps.mRNA_to_transcript || {{}};
  const c5 = steps.clip_5prime || {{}};

  const hasFixMeta = !!(gf.applied || mrna.applied || c5.applied);
  if (!hasFixMeta) return;

  let meta = '<div class="args-meta"><ul>';
  if (mrna.applied) meta += '<li>mRNA features were renamed to transcript features.</li>';
  if (gf.applied) {{
    const nAdded = Number(gf.n_genes_added || 0).toLocaleString('en-US');
    const fileLabel = gf.gene_ids_file ? ` (list: ${{escHtml(gf.gene_ids_file)}})` : '';
    meta += `<li>Added gene features: <strong>${{nAdded}}</strong>${{fileLabel}}</li>`;
  }}
  if (c5.applied) {{
    const nEv = Number(c5.n_events || 0).toLocaleString('en-US');
    const nGenes = Number(c5.n_genes_clipped || 0).toLocaleString('en-US');
    const logLabel = c5.log_file ? ` (log: ${{escHtml(c5.log_file)}})` : '';
    const listLabel = c5.gene_ids_file ? ` (genes: ${{escHtml(c5.gene_ids_file)}})` : '';
    meta += `<li>5' overlap clipping: <strong>${{nEv}}</strong> overlap events across <strong>${{nGenes}}</strong> genes${{logLabel}}${{listLabel}}</li>`;
  }}
  meta += '</ul></div>';

  document.getElementById('genomeFixSection').innerHTML = `
    <p class="section-title">Genome Fix Summary</p>
    <div class="args-box widget-card widget-accent-green">${{meta}}</div>`;
}})();

// ── Command-line args + rerun/log metadata ────────────────────────────────
(function buildArgs() {{
  document.getElementById('argsSection').innerHTML = `
    <p class="section-title">Command Used</p>
    <div class="args-box widget-card widget-accent-blue"><pre>${{escHtml(S.run_args || 'N/A')}}</pre></div>`;
}})();

// ── Execution summary (rerun/log metadata) ───────────────────────────────
(function buildExecutionSummary() {{
  const sections = D.log_sections || [];
  const notes = D.log_notes || [];
  const F = D.fix_info || {{}};
  const skipped = Array.isArray(F.skipped_steps) ? F.skipped_steps : [];
  const rerunMode = !!F.rerun_mode;
  const hasSkipMeta = rerunMode && skipped.length > 0;
  const hasMeta = hasSkipMeta || !!S.log_file || sections.length || notes.length || rerunMode;
  if (!hasMeta) return;

  let meta = '<div class="args-meta">';
  if (rerunMode) {{
    if (skipped.length) {{
      meta += '<p><strong>Rerun mode:</strong> skipped cached steps</p><ul>';
      skipped.forEach(x => {{
        meta += `<li>${{escHtml(String(x))}}</li>`;
      }});
      meta += '</ul>';
    }} else {{
      meta += '<p><strong>Rerun mode:</strong> enabled, no cached steps were skipped.</p>';
    }}
  }}
  if (S.log_file) meta += `<p><strong>Text log file:</strong> ${{escHtml(S.log_file)}}</p>`;
  if (sections.length) meta += `<p><strong>Log stages:</strong> ${{escHtml(sections.join(' | '))}}</p>`;
  meta += '</div>';

  document.getElementById('executionSummarySection').innerHTML = `
    <p class="section-title">Execution Summary</p>
    <div class="args-box widget-card widget-accent-teal">${{meta}}</div>`;
}})();
</script>
</body>
</html>"""


def _safe_json_for_html(data: dict) -> str:
    """
    Dump JSON and escape characters that could terminate a <script> block.
    """
    payload_json = json.dumps(data, indent=None, separators=(",", ":"))
    return (
        payload_json
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\\u2028", "\\\\u2028")
        .replace("\\u2029", "\\\\u2029")
    )
