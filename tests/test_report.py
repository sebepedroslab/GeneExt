"""
Tests for geneext/report.py

Covers:
  - _histogram          (bin computation)
  - _log10_histogram    (unified log10 bins)
  - _parse_mapping_stats
  - generate_html_report  (integration: writes valid HTML with correct data)

No BAM/samtools required.
"""

import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geneext.report import (
    _histogram,
    _log10_histogram,
    _parse_mapping_stats,
    _read_bed_col,
    _safe_json_for_html,
    generate_html_report,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tsv(rows, suffix=".tsv"):
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, newline=""
    )
    for row in rows:
        f.write("\t".join(str(x) for x in row) + "\n")
    f.close()
    return f.name


def _write_text(content, suffix=".txt"):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return f.name


def _make_tmpdir_with_data(
    n_ext=50,
    n_genic=80,
    n_noov=120,
    include_mapping_stats=False,
    seed=42,
):
    """Return a tmpdir path populated with synthetic pipeline output files."""
    rng = np.random.default_rng(seed)
    tmpdir = tempfile.mkdtemp(prefix="geneext_test_")

    # extensions.tsv
    ext_rows = [
        (f"gene_{i}", f"peak_{i}", int(abs(rng.lognormal(6.5, 0.8))))
        for i in range(n_ext)
    ]
    pd.DataFrame(ext_rows).to_csv(
        os.path.join(tmpdir, "extensions.tsv"), sep="\t", index=False, header=False
    )

    # genic_peaks.bed
    genic_cov = np.abs(rng.lognormal(3.8, 1.1, n_genic))
    _build_bed(os.path.join(tmpdir, "genic_peaks.bed"), genic_cov, rng)

    # allpeaks_noov.bed
    noov_cov = np.abs(rng.lognormal(2.5, 1.2, n_noov))
    _build_bed(os.path.join(tmpdir, "allpeaks_noov.bed"), noov_cov, rng)

    if include_mapping_stats:
        stats_txt = (
            "/path/before.gtf:\n"
            "Total reads: 1000000\n"
            "Mapped reads: 920000 (total: 92.0 %)\n"
            "Genic reads: 700000 (total: 70.0 %; mapped: 76.09 %)\n"
            "Orphan peaks: 20000 (total: 2.0 %; mapped: 2.17 %)\n"
            "Intergenic reads: 200000 (total: 20.0 %; mapped: 21.74 %)\n"
            "\n"
            "/path/after.gtf:\n"
            "Total reads: 1000000\n"
            "Mapped reads: 920000 (total: 92.0 %)\n"
            "Genic reads: 780000 (total: 78.0 %; mapped: 84.78 %)\n"
            "Orphan peaks: 20000 (total: 2.0 %; mapped: 2.17 %)\n"
            "Intergenic reads: 120000 (total: 12.0 %; mapped: 13.04 %)\n"
        )
        with open(os.path.join(tmpdir, "mapping_stats.txt"), "w") as fh:
            fh.write(stats_txt)

    return tmpdir


def _build_bed(path, coverages, rng):
    rows = []
    for i, cov in enumerate(coverages):
        rows.append(("chr1", i * 200, i * 200 + 100, f"pk_{i}", 0, "+", cov))
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, header=False)


# ---------------------------------------------------------------------------
# _histogram
# ---------------------------------------------------------------------------

class TestHistogram:
    def test_empty_input(self):
        labels, counts = _histogram([], n_bins=10)
        assert labels == [] and counts == []

    def test_returns_n_bins(self):
        labels, counts = _histogram(list(range(100)), n_bins=20)
        assert len(labels) == 20
        assert len(counts) == 20

    def test_counts_sum_to_total(self):
        data = list(range(200))
        _, counts = _histogram(data, n_bins=10)
        assert sum(counts) == len(data)

    def test_single_value(self):
        labels, counts = _histogram([42.0], n_bins=5)
        assert sum(counts) == 1

    def test_labels_are_strings(self):
        labels, _ = _histogram([1, 2, 3, 4, 5], n_bins=3)
        assert all(isinstance(l, str) for l in labels)


# ---------------------------------------------------------------------------
# _log10_histogram
# ---------------------------------------------------------------------------

class TestLog10Histogram:
    def test_empty_both(self):
        labels, g, n = _log10_histogram([], [])
        assert labels == [] and g == [] and n == []

    def test_zeros_ignored(self):
        # zeros and negatives should not crash
        labels, g, n = _log10_histogram([0, -1, 5, 10], [0, 3, 7])
        assert sum(g) == 2  # only 5, 10 are positive
        assert sum(n) == 2  # only 3, 7

    def test_shared_bins(self):
        genic = [1, 10, 100, 1000]
        noov  = [5, 50, 500]
        labels, g_counts, n_counts = _log10_histogram(genic, noov, n_bins=10)
        assert len(labels) == len(g_counts) == len(n_counts)

    def test_counts_sum_correctly(self):
        genic = list(range(1, 51))
        noov  = list(range(1, 101))
        _, g, n = _log10_histogram(genic, noov, n_bins=20)
        assert sum(g) == 50
        assert sum(n) == 100


# ---------------------------------------------------------------------------
# _read_bed_col
# ---------------------------------------------------------------------------

class TestReadBedCol:
    def test_missing_file(self):
        assert _read_bed_col("/nonexistent/file.bed", 6) == []

    def test_reads_correct_column(self):
        path = _write_tsv([
            ("chr1", 100, 200, "p1", 0, "+", 3.14),
            ("chr1", 300, 400, "p2", 0, "+", 6.28),
        ], suffix=".bed")
        try:
            vals = _read_bed_col(path, 6)
            assert len(vals) == 2
            assert abs(vals[0] - 3.14) < 1e-6
        finally:
            os.unlink(path)

    def test_col_out_of_range_returns_empty(self):
        path = _write_tsv([("chr1", 100, 200)], suffix=".bed")
        try:
            assert _read_bed_col(path, 10) == []
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# _parse_mapping_stats
# ---------------------------------------------------------------------------

class TestParseMappingStats:
    def test_missing_file(self):
        assert _parse_mapping_stats("/nonexistent.txt") == []

    def test_parses_two_sections(self):
        path = _write_text(
            "/path/before.gtf:\n"
            "Total reads: 1000\n"
            "Mapped reads: 900 (total: 90.0 %)\n"
            "Genic reads: 700 (total: 70.0 %; mapped: 77.78 %)\n"
            "Orphan peaks: 10 (total: 1.0 %; mapped: 1.11 %)\n"
            "Intergenic reads: 190 (total: 19.0 %; mapped: 21.11 %)\n"
            "\n"
            "/path/after.gtf:\n"
            "Total reads: 1000\n"
            "Mapped reads: 900 (total: 90.0 %)\n"
            "Genic reads: 800 (total: 80.0 %; mapped: 88.89 %)\n"
            "Orphan peaks: 10 (total: 1.0 %; mapped: 1.11 %)\n"
            "Intergenic reads: 90 (total: 9.0 %; mapped: 10.0 %)\n"
        )
        try:
            stats = _parse_mapping_stats(path)
            assert len(stats) == 2
            assert stats[0]["total"] == 1000
            assert stats[0]["genic"] == 700
            assert stats[0]["intergenic_pct"] == 19.0
            assert stats[1]["intergenic_pct"] == 9.0
        finally:
            os.unlink(path)

    def test_partial_file_returns_one_record(self):
        path = _write_text(
            "/only/one.gtf:\n"
            "Total reads: 500\n"
            "Mapped reads: 450 (total: 90.0 %)\n"
        )
        try:
            stats = _parse_mapping_stats(path)
            assert len(stats) == 1
            assert stats[0]["mapped"] == 450
        finally:
            os.unlink(path)

    def test_parses_basename_header(self):
        path = _write_text(
            "before.gtf:\n"
            "Total reads: 1000\n"
            "Mapped reads: 900 (total: 90.0 %)\n"
        )
        try:
            stats = _parse_mapping_stats(path)
            assert len(stats) == 1
            assert stats[0]["label"] == "before.gtf"
            assert stats[0]["mapped_pct"] == 90.0
        finally:
            os.unlink(path)



# ---------------------------------------------------------------------------
# _safe_json_for_html
# ---------------------------------------------------------------------------

class TestSafeJsonForHtml:
    def test_escapes_script_breakers(self):
        payload = {"x": "</script><script>alert(1)</script>", "y": "a&b"}
        out = _safe_json_for_html(payload)
        assert "</script>" not in out
        assert "\\u003c/script\\u003e" in out
        assert "\\u0026" in out


# ---------------------------------------------------------------------------
# generate_html_report  (integration)
# ---------------------------------------------------------------------------

class TestGenerateHtmlReport:
    def _run(self, **kwargs):
        defaults = dict(
            genefile="test_data/annotation.gtf",
            infmt="gtf",
            coverage_percentile=25,
            count_threshold=3.14,
            do_estimate=False,
            n_genes=500,
            run_args="geneext.py -g annotation.gtf -b alignments.bam -o out.gtf",
        )
        defaults.update(kwargs)
        tmpdir = _make_tmpdir_with_data(**{
            k: v for k, v in [
                ("n_ext", defaults.pop("n_ext", 50)),
                ("n_genic", defaults.pop("n_genic", 80)),
                ("n_noov", defaults.pop("n_noov", 120)),
                ("include_mapping_stats", defaults.get("do_estimate", False)),
            ]
        })
        output_base = tempfile.mktemp(suffix=".gtf")
        return generate_html_report(tempdir=tmpdir, outputfile=output_base, **defaults), output_base

    def test_creates_file(self):
        path, _ = self._run()
        assert os.path.exists(path)
        assert path.endswith(".Report.html")

    def test_file_is_non_empty(self):
        path, _ = self._run()
        assert os.path.getsize(path) > 5000

    def test_valid_html(self):
        path, _ = self._run()
        html = open(path).read()
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_stats_embedded_in_json(self):
        path, _ = self._run(n_genes=874)
        html = open(path).read()
        assert '"n_genes":874' in html or '"n_genes": 874' in html

    def test_extension_histogram_data_present(self):
        path, _ = self._run(n_ext=50)
        html = open(path).read()
        assert '"ext_hist"' in html
        assert '"counts"' in html

    def test_coverage_histogram_data_present(self):
        path, _ = self._run(n_genic=40, n_noov=60)
        html = open(path).read()
        assert '"cov_hist"' in html
        assert '"counts_genic"' in html
        assert '"counts_noov"' in html

    def test_log10_threshold_embedded(self):
        path, _ = self._run(count_threshold=10.0)
        html = open(path).read()
        assert f'"log10_threshold":{math.log10(10.0)}' in html or "log10_threshold" in html

    def test_run_args_embedded(self):
        path, _ = self._run(run_args="geneext.py -g genome.gtf -b reads.bam")
        html = open(path).read()
        assert "reads.bam" in html

    def test_zero_coverage_percentile_handled(self):
        """coverage_percentile=0 means filtering was disabled — should not crash."""
        path, _ = self._run(coverage_percentile=0, count_threshold=None)
        assert os.path.exists(path)

    def test_empty_extensions_handled(self):
        """No extended genes — should still produce a valid page."""
        tmpdir = _make_tmpdir_with_data(n_ext=0)
        output_base = tempfile.mktemp(suffix=".gtf")
        path = generate_html_report(
            tempdir=tmpdir,
            outputfile=output_base,
            genefile="test_data/annotation.gtf",
            infmt="gtf",
            coverage_percentile=25,
            n_genes=200,
        )
        html = open(path).read()
        assert "<!DOCTYPE html>" in html

    def test_mapping_stats_section_present_when_estimate(self):
        tmpdir = _make_tmpdir_with_data(n_ext=30, include_mapping_stats=True)
        output_base = tempfile.mktemp(suffix=".gtf")
        path = generate_html_report(
            tempdir=tmpdir,
            outputfile=output_base,
            genefile="test_data/annotation.gtf",
            infmt="gtf",
            coverage_percentile=25,
            do_estimate=True,
            n_genes=400,
        )
        html = open(path).read()
        assert "mapping_stats" in html
        assert "Intergenic" in html

    def test_mapping_stats_section_absent_without_estimate(self):
        tmpdir = _make_tmpdir_with_data(n_ext=30, include_mapping_stats=False)
        output_base = tempfile.mktemp(suffix=".gtf")
        path = generate_html_report(
            tempdir=tmpdir,
            outputfile=output_base,
            genefile="test_data/annotation.gtf",
            infmt="gtf",
            coverage_percentile=25,
            do_estimate=False,
            n_genes=400,
        )
        html = open(path).read()
        # mapping_stats key will be present in JSON but empty list
        assert '"mapping_stats":[]' in html or '"mapping_stats": []' in html


    def test_orphan_peak_count_ignores_comments_and_trailing_newline(self):
        tmpdir = _make_tmpdir_with_data(n_ext=10)
        orphan_path = os.path.join(tmpdir, "orphan_merged.bed")
        with open(orphan_path, "w") as fh:
            fh.write("# comment\n")
            fh.write("chr1\t10\t20\torph1\n")
            fh.write("chr1\t30\t40\torph2")

        output_base = tempfile.mktemp(suffix=".gtf")
        path = generate_html_report(
            tempdir=tmpdir,
            outputfile=output_base,
            genefile="test_data/annotation.gtf",
            infmt="gtf",
            coverage_percentile=25,
            do_estimate=False,
            n_genes=400,
        )
        html = open(path).read()
        assert '"n_orphan_peaks":2' in html or '"n_orphan_peaks": 2' in html

    def test_fix_info_includes_clip_gene_list_path(self):
        tmpdir = _make_tmpdir_with_data(n_ext=10)
        fix_info = {
            "schema": "v1",
            "input_genefile": "input.gtf",
            "final_genefile": "genome.fixed.gtf",
            "steps": {
                "mRNA_to_transcript": {"applied": False},
                "gene_features_added": {
                    "applied": True,
                    "n_genes_added": 3,
                    "gene_ids_file": "/tmp/fixed_genes_added.txt",
                },
                "clip_5prime": {
                    "applied": True,
                    "n_events": 12,
                    "n_genes_clipped": 7,
                    "log_file": "/tmp/genome.fixed.gtf.5clip.log",
                    "gene_ids_file": "/tmp/fiveprime_clipped_genes.txt",
                },
            },
        }
        with open(os.path.join(tmpdir, "report_fix_info.json"), "w") as fh:
            import json
            json.dump(fix_info, fh)

        output_base = tempfile.mktemp(suffix=".gtf")
        path = generate_html_report(
            tempdir=tmpdir,
            outputfile=output_base,
            genefile="test_data/annotation.gtf",
            infmt="gtf",
            coverage_percentile=25,
            do_estimate=False,
            n_genes=400,
        )
        html = open(path).read()
        assert "fiveprime_clipped_genes.txt" in html

    def test_rerun_skipped_steps_are_embedded(self):
        tmpdir = _make_tmpdir_with_data(n_ext=10)
        fix_info = {
            "schema": "v1",
            "input_genefile": "input.gtf",
            "final_genefile": "genome.fixed.gtf",
            "rerun_mode": True,
            "force_mode": False,
            "skipped_steps": [
                "MACS2 peak calling (reused allpeaks.bed)",
                "Peak filtering (reused cached coverage/filter outputs)",
            ],
            "steps": {
                "mRNA_to_transcript": {"applied": False},
                "gene_features_added": {"applied": False, "n_genes_added": 0, "gene_ids_file": ""},
                "clip_5prime": {"applied": False, "n_events": 0, "n_genes_clipped": 0, "log_file": "", "gene_ids_file": ""},
            },
        }
        with open(os.path.join(tmpdir, "report_fix_info.json"), "w") as fh:
            import json
            json.dump(fix_info, fh)

        output_base = tempfile.mktemp(suffix=".gtf")
        path = generate_html_report(
            tempdir=tmpdir,
            outputfile=output_base,
            genefile="test_data/annotation.gtf",
            infmt="gtf",
            coverage_percentile=25,
            do_estimate=False,
            n_genes=400,
        )
        html = open(path).read()
        assert "Rerun mode" in html
        assert "reused allpeaks.bed" in html

    def test_extension_param_section_for_user_value(self):
        tmpdir = _make_tmpdir_with_data(n_ext=10)
        fix_info = {
            "schema": "v1",
            "input_genefile": "input.gtf",
            "final_genefile": "output.gtf",
            "extension_param": {
                "name": "--maxdist",
                "mode": "user",
                "user_value_bp": 1200,
                "effective_value_bp": 1200,
                "auto_quantile": None,
            },
            "steps": {
                "mRNA_to_transcript": {"applied": False},
                "gene_features_added": {"applied": False, "n_genes_added": 0, "gene_ids_file": ""},
                "clip_5prime": {"applied": False, "n_events": 0, "n_genes_clipped": 0, "log_file": "", "gene_ids_file": ""},
            },
        }
        with open(os.path.join(tmpdir, "report_fix_info.json"), "w") as fh:
            import json
            json.dump(fix_info, fh)

        output_base = tempfile.mktemp(suffix=".gtf")
        path = generate_html_report(
            tempdir=tmpdir,
            outputfile=output_base,
            genefile="test_data/annotation.gtf",
            infmt="gtf",
            coverage_percentile=25,
            do_estimate=False,
            n_genes=400,
        )
        html = open(path).read()
        assert "Extension Parameter" in html
        assert "User-defined" in html
        assert '"mode":"user"' in html
        assert '"effective_value_bp":1200' in html

    def test_extension_param_section_for_auto_value(self):
        tmpdir = _make_tmpdir_with_data(n_ext=10)
        fix_info = {
            "schema": "v1",
            "input_genefile": "input.gtf",
            "final_genefile": "output.gtf",
            "extension_param": {
                "name": "--maxdist",
                "mode": "auto",
                "user_value_bp": None,
                "effective_value_bp": 845,
                "auto_quantile": 0.5,
            },
            "steps": {
                "mRNA_to_transcript": {"applied": False},
                "gene_features_added": {"applied": False, "n_genes_added": 0, "gene_ids_file": ""},
                "clip_5prime": {"applied": False, "n_events": 0, "n_genes_clipped": 0, "log_file": "", "gene_ids_file": ""},
            },
        }
        with open(os.path.join(tmpdir, "report_fix_info.json"), "w") as fh:
            import json
            json.dump(fix_info, fh)

        output_base = tempfile.mktemp(suffix=".gtf")
        path = generate_html_report(
            tempdir=tmpdir,
            outputfile=output_base,
            genefile="test_data/annotation.gtf",
            infmt="gtf",
            coverage_percentile=25,
            do_estimate=False,
            n_genes=400,
        )
        html = open(path).read()
        assert "Extension Parameter" in html
        assert "Auto-estimated" in html
        assert '"mode":"auto"' in html
        assert '"auto_quantile":0.5' in html
        assert '"effective_value_bp":845' in html

    def test_peak_flow_payload_and_section_present(self):
        tmpdir = _make_tmpdir_with_data(n_ext=10)
        with open(os.path.join(tmpdir, "allpeaks.bed"), "w") as fh:
            fh.write("chr1\t10\t20\tp1\nchr1\t30\t40\tp2\nchr1\t50\t60\tp3\n")
        with open(os.path.join(tmpdir, "allpeaks_noov_fcov.bed"), "w") as fh:
            fh.write("chr1\t10\t20\tp1\nchr1\t30\t40\tp2\n")
        with open(os.path.join(tmpdir, "orphan_merged.bed"), "w") as fh:
            fh.write("chr1\t100\t120\torph1\n")

        output_base = tempfile.mktemp(suffix=".gtf")
        path = generate_html_report(
            tempdir=tmpdir,
            outputfile=output_base,
            genefile="test_data/annotation.gtf",
            infmt="gtf",
            coverage_percentile=25,
            do_estimate=False,
            n_genes=400,
        )
        html = open(path).read()
        assert "Peak Filtering Flow" in html
        assert '"peak_flow"' in html
        assert '"initial_called":3' in html
        assert '"passed_filtering":2' in html
