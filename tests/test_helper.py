"""
Tests for geneext/helper.py

Run with:  python -m pytest tests/ -v

Integration tests use the real test_data/ files so they exercise the same
code paths that are exercised during a real pipeline run.
"""

import os
import sys
import tempfile
import textwrap

import pytest

# Make sure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geneext.helper import (
    Region,
    add_gene_features,
    get_prefixed_path,
    get_extension,
    parse_bed,
    parse_gtf,
    guess_format_fromfile,
    compute_mean_coverage,
    get_coverage_percentile,
    filter_by_coverage,
    estimate_mapping,
    get_chr_sizes,
)

# ---------------------------------------------------------------------------
# Paths to bundled test data
# ---------------------------------------------------------------------------

TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data")
TEST_GTF = os.path.join(TEST_DIR, "annotation.gtf")
TEST_BAM = os.path.join(TEST_DIR, "alignments.bam")
TEST_BAI = os.path.join(TEST_DIR, "alignments.bam.bai")

import shutil

HAS_TEST_DATA = os.path.exists(TEST_GTF) and os.path.exists(TEST_BAM)
HAS_SAMTOOLS = shutil.which("samtools") is not None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp(content, suffix=""):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return f.name


def _tmp_bed(lines):
    return _tmp(lines, suffix=".bed")


# ---------------------------------------------------------------------------
# Region – construction
# ---------------------------------------------------------------------------

class TestRegionConstruction:
    def test_basic_fields(self):
        r = Region(chrom="chr1", start=100, end=200, strand="+", id="gene1")
        assert r.chrom == "chr1"
        assert r.start == 100
        assert r.end == 200
        assert r.strand == "+"
        assert r.id == "gene1"

    def test_default_score_is_zero(self):
        r = Region(chrom="chr1", start=0, end=1, strand="+")
        assert r.score == 0

    def test_explicit_score(self):
        r = Region(chrom="chr1", start=0, end=1, strand="+", score=42)
        assert r.score == 42

    def test_length(self):
        r = Region(chrom="chr1", start=100, end=300, strand="+")
        assert r.length() == 200

    def test_zero_length(self):
        r = Region(chrom="chr1", start=50, end=50, strand="+")
        assert r.length() == 0


# ---------------------------------------------------------------------------
# Region – is_overlapping
# ---------------------------------------------------------------------------

class TestRegionIsOverlapping:
    def _r(self, chrom, start, end, strand="+"):
        return Region(chrom=chrom, start=start, end=end, strand=strand)

    def test_non_overlapping_left(self):
        assert not Region.is_overlapping(self._r("chr1", 100, 200), self._r("chr1", 300, 400))

    def test_non_overlapping_right(self):
        assert not Region.is_overlapping(self._r("chr1", 300, 400), self._r("chr1", 100, 200))

    def test_overlapping(self):
        assert Region.is_overlapping(self._r("chr1", 100, 300), self._r("chr1", 200, 400))

    def test_contained(self):
        assert Region.is_overlapping(self._r("chr1", 100, 400), self._r("chr1", 150, 250))

    def test_adjacent_treated_as_overlapping(self):
        # end == start: neither "end < start" condition triggers → reported overlapping
        assert Region.is_overlapping(self._r("chr1", 100, 200), self._r("chr1", 200, 300))


# ---------------------------------------------------------------------------
# Region – directional helpers
# ---------------------------------------------------------------------------

class TestRegionDirectional:
    def _r(self, chrom, start, end, strand):
        return Region(chrom=chrom, start=start, end=end, strand=strand)

    def test_downstream_plus(self):
        assert Region.a_is_downstream_b(self._r("chr1", 500, 600, "+"), self._r("chr1", 100, 200, "+"))

    def test_not_downstream_plus(self):
        assert Region.a_is_downstream_b(self._r("chr1", 100, 200, "+"), self._r("chr1", 500, 600, "+")) is False

    def test_downstream_minus(self):
        assert Region.a_is_downstream_b(self._r("chr1", 100, 200, "-"), self._r("chr1", 500, 600, "-"))

    def test_upstream_plus(self):
        assert Region.a_is_upstream_b(self._r("chr1", 100, 200, "+"), self._r("chr1", 500, 600, "+"))

    def test_upstream_minus(self):
        assert Region.a_is_upstream_b(self._r("chr1", 500, 600, "-"), self._r("chr1", 100, 200, "-"))

    def test_none_on_overlap(self):
        a = self._r("chr1", 100, 300, "+")
        b = self._r("chr1", 200, 400, "+")
        assert Region.a_is_downstream_b(a, b) is None
        assert Region.a_is_upstream_b(a, b) is None

    def test_none_different_strands(self):
        a = self._r("chr1", 100, 200, "+")
        b = self._r("chr1", 300, 400, "-")
        assert Region.a_is_downstream_b(a, b) is None
        assert Region.a_is_upstream_b(a, b) is None


# ---------------------------------------------------------------------------
# Region – get_distance  (Bug 1 fixed: operator-precedence on strand check)
# ---------------------------------------------------------------------------

class TestRegionGetDistance:
    def _r(self, chrom, start, end, strand):
        return Region(chrom=chrom, start=start, end=end, strand=strand)

    def test_same_plus_strand(self):
        # For '+': distance = |end_a - end_b| = |200 - 500| = 300
        d = Region.get_distance(self._r("chr1", 100, 200, "+"), self._r("chr1", 400, 500, "+"))
        assert d == 300

    def test_same_minus_strand(self):
        # For '-': distance = |start_a - start_b| = |100 - 400| = 300
        d = Region.get_distance(self._r("chr1", 100, 200, "-"), self._r("chr1", 400, 500, "-"))
        assert d == 300

    def test_none_for_overlapping(self):
        assert Region.get_distance(
            self._r("chr1", 100, 300, "+"), self._r("chr1", 200, 400, "+")
        ) is None

    def test_none_different_chromosomes(self):
        assert Region.get_distance(
            self._r("chr1", 100, 200, "+"), self._r("chr2", 300, 400, "+")
        ) is None

    def test_none_mixed_strands(self):
        """After Bug 1 fix, mixed-strand pairs must return None."""
        d = Region.get_distance(self._r("chr1", 100, 200, "-"), self._r("chr1", 400, 500, "+"))
        assert d is None, (
            "get_distance should return None for mixed-strand regions (Bug 1 fix regression)"
        )

    def test_none_mixed_strands_reversed(self):
        d = Region.get_distance(self._r("chr1", 100, 200, "+"), self._r("chr1", 400, 500, "-"))
        assert d is None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

class TestGetPrefixedPath:
    def test_simple(self):
        assert get_prefixed_path("/some/dir/file.txt") == "/some/dir/tmp_file.txt"

    def test_custom_prefix(self):
        assert get_prefixed_path("/dir/file.bed", prefix="bak_") == "/dir/bak_file.bed"

    def test_no_extension(self):
        assert get_prefixed_path("/dir/noext") == "/dir/tmp_noext"

    def test_root_file(self):
        assert get_prefixed_path("file.gtf") == "tmp_file.gtf"


class TestGetExtension:
    def test_gtf(self):
        assert get_extension("annotation.gtf") == "gtf"

    def test_bam(self):
        assert get_extension("/path/to/reads.bam") == "bam"

    def test_bed(self):
        assert get_extension("peaks.bed") == "bed"

    def test_no_ext(self):
        assert get_extension("noextension") == ""


# ---------------------------------------------------------------------------
# parse_bed
# ---------------------------------------------------------------------------

BED_CONTENT = textwrap.dedent("""\
    chr1\t100\t200\tpeak1\t0\t+
    chr1\t300\t400\tpeak2\t0\t-
    chr2\t500\t600\tpeak3\t0\t+
""")

class TestParseBed:
    def test_basic_parse(self):
        path = _tmp_bed(BED_CONTENT)
        try:
            regs = parse_bed(path)
            assert len(regs) == 3
        finally:
            os.unlink(path)

    def test_fields(self):
        path = _tmp_bed(BED_CONTENT)
        try:
            r = parse_bed(path)[0]
            assert r.chrom == "chr1"
            assert r.start == 100
            assert r.end == 200
            assert r.id == "peak1"
            assert r.strand == "+"
        finally:
            os.unlink(path)

    def test_comment_lines_skipped(self):
        path = _tmp_bed("# comment\n" + BED_CONTENT)
        try:
            assert len(parse_bed(path)) == 3
        finally:
            os.unlink(path)

    def test_empty_returns_empty_list(self):
        path = _tmp_bed("")
        try:
            assert parse_bed(path) == []
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# parse_gtf – integration tests using test_data/annotation.gtf
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TEST_DATA, reason="test_data/ not found")
class TestParseGtfIntegration:
    def test_parse_returns_list(self):
        regs = parse_gtf(TEST_GTF)
        assert isinstance(regs, list) and len(regs) > 0

    def test_parse_genes_only(self):
        regs = parse_gtf(TEST_GTF, featuretype="gene")
        assert all(isinstance(r, Region) for r in regs)

    def test_unknown_featuretype_raises(self):
        with pytest.raises(ValueError):
            parse_gtf(TEST_GTF, featuretype="nonexistent_feature_xyz")

    def test_region_fields_populated(self):
        r = parse_gtf(TEST_GTF, featuretype="gene")[0]
        assert r.chrom and r.start >= 0 and r.end > r.start and r.strand in ("+", "-", ".")


# ---------------------------------------------------------------------------
# guess_format_fromfile – integration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TEST_DATA, reason="test_data/ not found")
class TestGuessFormatIntegration:
    def test_detects_gtf(self):
        assert guess_format_fromfile(TEST_GTF) == "gtf"

    def test_detects_bed(self):
        path = _tmp("chr1\t100\t200\tpeak1\t0\t+\n", suffix=".bed")
        try:
            assert guess_format_fromfile(path) == "bed"
        finally:
            os.unlink(path)

    def test_skips_comment_lines(self):
        path = _tmp("# comment\nchr1\t100\t200\tpeak1\t0\t+\n", suffix=".bed")
        try:
            assert guess_format_fromfile(path) == "bed"
        finally:
            os.unlink(path)

    def test_raises_on_unknown_format(self):
        path = _tmp("col1\tcol2\n", suffix=".txt")
        try:
            with pytest.raises(ValueError):
                guess_format_fromfile(path)
        finally:
            os.unlink(path)


class TestAddGeneFeatures:
    def test_returns_added_gene_ids_for_gtf_without_genes(self):
        gtf_in = _tmp(
            "\n".join(
                [
                    'chr1\t.\ttranscript\t100\t200\t.\t+\t.\ttranscript_id "t1"; gene_id "g1";',
                    'chr1\t.\texon\t100\t120\t.\t+\t.\ttranscript_id "t1"; gene_id "g1";',
                    'chr1\t.\ttranscript\t300\t450\t.\t-\t.\ttranscript_id "t2"; gene_id "g2";',
                    'chr1\t.\texon\t400\t450\t.\t-\t.\ttranscript_id "t2"; gene_id "g2";',
                    "",
                ]
            ),
            suffix=".gtf",
        )
        gtf_out = tempfile.mktemp(suffix=".gtf")
        try:
            added = add_gene_features(
                infile=gtf_in,
                outfile=gtf_out,
                infmt="gtf",
                verbose=False,
            )
            assert isinstance(added, list)
            assert set(added) == {"g1", "g2"}
            out_txt = open(gtf_out).read()
            assert '\tgene\t' in out_txt
        finally:
            if os.path.exists(gtf_in):
                os.unlink(gtf_in)
            if os.path.exists(gtf_out):
                os.unlink(gtf_out)


# ---------------------------------------------------------------------------
# compute_mean_coverage  (Bug 4 fixed: zero-length guard)
# ---------------------------------------------------------------------------

class FakeAln:
    """Minimal pysam.AlignmentFile stub."""
    def __init__(self, count=50):
        self._count = count

    def count(self, **kwargs):
        return self._count


class TestComputeMeanCoverage:
    def test_normal_region(self):
        r = Region(chrom="chr1", start=100, end=200, strand="+")
        assert compute_mean_coverage(r, FakeAln(50)) == pytest.approx(0.5)

    def test_zero_length_returns_zero(self):
        """Bug 4 fix: zero-length region must return 0.0, not raise ZeroDivisionError."""
        r = Region(chrom="chr1", start=100, end=100, strand="+")
        assert compute_mean_coverage(r, FakeAln(5)) == 0.0


# ---------------------------------------------------------------------------
# get_coverage_percentile  (Bug 5 fixed: empty file handled gracefully)
# ---------------------------------------------------------------------------

def _cov_file(values):
    lines = [f"chr1\t{i*100}\t{i*100+50}\tpeak{i}\t0\t+\t{v}" for i, v in enumerate(values)]
    return _tmp("\n".join(lines) + "\n", suffix=".bed")


class TestGetCoveragePercentile:
    def test_zero_percentile(self):
        path = _cov_file([10, 20, 30])
        try:
            assert get_coverage_percentile(inputfile=path, percentile=0) == "0"
        finally:
            os.unlink(path)

    def test_negative_percentile(self):
        path = _cov_file([10, 20, 30])
        try:
            assert get_coverage_percentile(inputfile=path, percentile=-5) == "0"
        finally:
            os.unlink(path)

    def test_50th_percentile(self):
        path = _cov_file([10, 20, 30, 40, 50])
        try:
            assert float(get_coverage_percentile(inputfile=path, percentile=50)) == pytest.approx(30.0, abs=1.0)
        finally:
            os.unlink(path)

    def test_100th_percentile(self):
        path = _cov_file([10, 20, 30, 40, 50])
        try:
            assert float(get_coverage_percentile(inputfile=path, percentile=100)) == pytest.approx(50.0, abs=1.0)
        finally:
            os.unlink(path)

    def test_returns_string(self):
        path = _cov_file([1, 2, 3])
        try:
            assert isinstance(get_coverage_percentile(inputfile=path, percentile=50), str)
        finally:
            os.unlink(path)

    def test_empty_file_returns_zero(self):
        """Bug 5 fix: empty file must return '0', not raise EmptyDataError."""
        path = _tmp("", suffix=".bed")
        try:
            assert get_coverage_percentile(inputfile=path, percentile=50) == "0"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# filter_by_coverage  (Bug 6 fixed: empty-input division-by-zero)
# ---------------------------------------------------------------------------

def _cov_bed_file(rows):
    lines = ["\t".join(str(x) for x in row) for row in rows]
    return _tmp("\n".join(lines) + "\n", suffix=".bed")


class TestFilterByCoverage:
    def test_filters_below_threshold(self):
        rows = [("chr1", 0, 100, "p1", 0, "+", 5),
                ("chr1", 200, 300, "p2", 0, "+", 15),
                ("chr1", 400, 500, "p3", 0, "+", 25)]
        infile = _cov_bed_file(rows)
        outfile = tempfile.mktemp(suffix=".bed")
        try:
            filter_by_coverage(inputfile=infile, outputfile=outfile, threshold=10, do_message=False)
            with open(outfile) as fh:
                lines = [l for l in fh if l.strip()]
            assert len(lines) == 2
        finally:
            os.unlink(infile)
            if os.path.exists(outfile):
                os.unlink(outfile)

    def test_all_filtered_out(self):
        infile = _cov_bed_file([("chr1", 0, 100, "p1", 0, "+", 1)])
        outfile = tempfile.mktemp(suffix=".bed")
        try:
            filter_by_coverage(inputfile=infile, outputfile=outfile, threshold=100, do_message=False)
            with open(outfile) as fh:
                assert [l for l in fh if l.strip()] == []
        finally:
            os.unlink(infile)
            if os.path.exists(outfile):
                os.unlink(outfile)

    def test_do_message_empty_input_no_crash(self):
        """Bug 6 fix: empty input with do_message=True must not raise ZeroDivisionError."""
        infile = _tmp("", suffix=".bed")
        outfile = tempfile.mktemp(suffix=".bed")
        try:
            # Should not raise; returns a message string or None
            result = filter_by_coverage(
                inputfile=infile, outputfile=outfile, threshold=10, do_message=True
            )
            # result is the message string; 0/0 case must not crash
            assert result is not None
        finally:
            os.unlink(infile)
            if os.path.exists(outfile):
                os.unlink(outfile)

    def test_do_message_returns_string(self):
        rows = [("chr1", 0, 100, "p1", 0, "+", 20)]
        infile = _cov_bed_file(rows)
        outfile = tempfile.mktemp(suffix=".bed")
        try:
            msg = filter_by_coverage(
                inputfile=infile, outputfile=outfile, threshold=10, do_message=True
            )
            assert isinstance(msg, str) and "Retained" in msg
        finally:
            os.unlink(infile)
            if os.path.exists(outfile):
                os.unlink(outfile)


# ---------------------------------------------------------------------------
# estimate_mapping – Bug 3 fix: function must return a tuple
# ---------------------------------------------------------------------------

class TestEstimateMappingSignature:
    def test_returns_four_values_with_real_bam(self):
        """
        Bug 3 fix: estimate_mapping must return (Ntot, Nmap, Ngen, Nigen).
        Uses the bundled test BAM and a simple genic BED derived from the GTF.
        """
        if not HAS_TEST_DATA:
            pytest.skip("test_data/ not found")
        if not HAS_SAMTOOLS:
            pytest.skip("samtools not in PATH")

        import subprocess
        # Build a minimal genic BED from the GTF
        genic_bed = tempfile.mktemp(suffix=".bed")
        try:
            cmd = (
                "awk -F'\\t' '$3==\"gene\"{print $1\"\\t\"$4\"\\t\"$5}' %s > %s"
                % (TEST_GTF, genic_bed)
            )
            subprocess.run(cmd, shell=True, check=True)

            result = estimate_mapping(
                bamfile=TEST_BAM,
                genicbed=genic_bed,
                intergenicbed=genic_bed,  # placeholder; not used in current impl
                threads=1,
            )
            assert result is not None, "estimate_mapping returned None (Bug 3 not fixed)"
            Ntot, Nmap, Ngen, Nigen = result
            assert Ntot >= 0
            assert Nmap >= 0
            assert Ngen >= 0
            assert Nigen >= 0
        finally:
            if os.path.exists(genic_bed):
                os.unlink(genic_bed)


# ---------------------------------------------------------------------------
# get_chr_sizes – Bug 2 fix: must NOT quit() after indexing
# ---------------------------------------------------------------------------

class TestGetChrSizesNoQuit:
    def test_does_not_quit_when_index_exists(self):
        """
        Bug 2 fix: when .bai exists, get_chr_sizes must complete normally
        and produce an outfile without calling quit().
        """
        if not HAS_TEST_DATA:
            pytest.skip("test_data/ not found")
        if not os.path.exists(TEST_BAI):
            pytest.skip("BAM index not present")
        if not HAS_SAMTOOLS:
            pytest.skip("samtools not in PATH")

        outfile = tempfile.mktemp(suffix=".txt")
        try:
            # Must not raise SystemExit
            get_chr_sizes(bamfile=TEST_BAM, outfile=outfile)
            assert os.path.exists(outfile), "outfile was not created"
            with open(outfile) as fh:
                lines = [l.strip() for l in fh if l.strip()]
            assert len(lines) > 0, "chromosome sizes file is empty"
        finally:
            if os.path.exists(outfile):
                os.unlink(outfile)

    def test_quit_not_called_after_indexing(self, monkeypatch):
        """
        Bug 2 fix regression: even when .bai is absent, the function must
        index the file and continue — not call quit().
        """
        import geneext.helper as helper_module

        monkeypatch.setattr(os.path, "exists", lambda p: False)
        monkeypatch.setattr(helper_module, "index_bam", lambda *a, **kw: None)

        outfile = tempfile.mktemp(suffix=".txt")
        try:
            # Should NOT raise SystemExit after the fix
            try:
                get_chr_sizes(bamfile="/fake/file.bam", outfile=outfile)
            except SystemExit:
                pytest.fail("get_chr_sizes still calls quit() after indexing (Bug 2 not fixed)")
        finally:
            if os.path.exists(outfile):
                os.unlink(outfile)
