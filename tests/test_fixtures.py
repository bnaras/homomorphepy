"""The fixture contracts, asserted from Python.

These tests need no crypto backend by design: the fixture layer is what
lets the Python twin be developed and tested where `openfhe` cannot be
installed, which currently includes macOS.

Reference values below were printed from R independently (17
significant digits) and are compared with `==`, not `approx`. The point
of the raw-float64 fixture format is that the bytes round-trip exactly;
an approximate assertion here would defeat it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from homomorphepy import (
    FixtureError,
    load_dlbcl,
    load_dlbcl_gex,
    load_golden,
    load_json,
    manifest,
    site_order,
    verify_all,
)


def test_manifest_declares_schema_and_provenance():
    m = manifest()
    assert m["schema_version"] == 1
    assert m["homomorpheR"]
    assert m["openfhe_R"]
    assert "rpois" in m["rng"]["note"]


def test_every_declared_fixture_matches_its_digest():
    digests = verify_all()
    assert len(digests) >= 10
    assert all(len(v) == 64 for v in digests.values())


class TestExpressionMatrix:
    """Bit-exactness, because the screen's rank-100 boundary depends on it."""

    def test_shape_and_dtype(self):
        gex, rows, cols = load_dlbcl_gex()
        assert gex.shape == (235, 6416)
        assert gex.dtype == np.dtype("float64")
        assert len(rows) == 235 and len(cols) == 6416

    def test_values_are_bit_identical_to_R(self):
        gex, _, _ = load_dlbcl_gex()
        assert list(gex[0, :3]) == [
            -0.221,
            -0.17860000000000001,
            -0.050250000000000003,
        ]
        assert gex[234, 6415] == 1.2190000000000001

    def test_column_sums_are_bit_identical(self):
        # colSums feeds the univariate Cox screen; a one-ulp difference
        # can reorder probes at the K=100 cutoff.
        gex, _, _ = load_dlbcl_gex()
        cs = gex.sum(axis=0)
        assert cs[0] == 7.7198723849372453
        assert cs[1] == 3.7857083682008352


class TestClinicalTable:
    def test_shape_and_forced_dtypes(self):
        df = load_dlbcl()
        assert df.shape == (235, 12)
        assert df["ID"].dtype == "int64"
        assert df["status"].dtype == "int64"
        assert isinstance(df["Subgroup"].dtype, pd.CategoricalDtype)

    def test_float_columns_round_trip_exactly(self):
        df = load_dlbcl()
        assert list(df["time"][:3]) == [4.0, 4.9000000000000004, 5.5999999999999996]

    def test_site_order_is_forced_not_alphabetical(self):
        # Site order is protocol semantics: index 0 is the lead
        # decryptor in the threshold decryption. Alphabetical sorting
        # would put ABC first and silently permute the protocol.
        order = site_order()
        assert order == ["GCB", "ABC", "Type III"]
        assert order != sorted(order)

    def test_categorical_survives_groupby(self):
        df = load_dlbcl()
        keys = list(df.groupby("Subgroup", observed=True, sort=True).groups)
        assert keys == site_order()

    def test_ids_align_with_expression_rownames(self):
        # The invariant the Cox-lasso example asserts before it starts.
        df = load_dlbcl()
        _, rows, _ = load_dlbcl_gex()
        assert [str(i) for i in df["ID"]] == rows


class TestSimulatedInputs:
    """Inputs R's RNG produced; numpy cannot regenerate any of them."""

    def test_mle_poisson_counts(self):
        f = load_json("mle_poisson")
        assert f["seed"] == 17822
        assert len(f["y"]) == 40
        assert sum(len(s["y"]) for s in f["sites"]) == 40

    def test_query_count_expectation_is_reproducible_from_the_data(self):
        # The exported expectation must follow from the exported rows,
        # or the fixture is internally inconsistent.
        f = load_json("query_count")
        recomputed = [
            int(
                (
                    (pd.DataFrame(s)["age"] < 50)
                    & (pd.DataFrame(s)["sex"] == "F")
                    & (pd.DataFrame(s)["bm"] < 0.2)
                ).sum()
            )
            for s in f["sites"]
        ]
        assert recomputed == f["expected"]["per_site"]
        assert sum(recomputed) == f["expected"]["total"]

    def test_query_count_is_declared_exact(self):
        # BFV is exact arithmetic: this example asserts ==, never a
        # tolerance, so the fixture must say so.
        f = load_json("query_count")
        assert f["expected"]["exact"] is True
        assert f["expected"]["plaintext_modulus"] == 65537

    def test_regression_training_set_is_aligned(self):
        f = load_json("encrypted_regression")
        assert len(f["age"]) == len(f["biomarker"]) == len(f["outcome"]) == 500

    def test_aggregation_site_sizes(self):
        f = load_json("aggregation_sites")
        assert [len(s["age"]) for s in f["sites"]] == [1000, 500, 1500]

    def test_secure_inference_panel_is_deterministic(self):
        f = load_json("secure_inference")
        assert f["seed"] is None
        assert all(len(v) == 8 for v in f["biomarkers"].values())


class TestGoldenOutputs:
    def test_structure(self):
        g = load_golden()
        assert g["params"]["K"] == 100
        assert len(g["top_idx"]) == 100
        assert g["n_iter_enc"] == g["n_iter_ref"] == 147
        assert len(g["trajectory"]) == 147
        assert len(g["trajectory"][0]) == 100

    def test_shipped_encrypted_fit_is_within_its_ckks_tolerance(self):
        g = load_golden()
        d = float(np.max(np.abs(np.array(g["z_enc"]) - np.array(g["z_ref"]))))
        assert g["tolerances"]["z_enc_vs_z_ref"].holds(d)

    def test_tolerance_ladder_separates_solver_from_crypto(self):
        # The whole point of the ladder: a cross-language solver
        # difference is orders of magnitude larger than CKKS noise, and
        # reporting one as the other misdiagnoses the failure.
        t = load_golden()["tolerances"]
        assert t["z_enc_vs_z_ref"].kind == "ckks"
        assert t["z_ref_vs_python"].kind == "statistical"
        assert t["top_idx"].kind == "set_equality"
        assert t["top_idx"].value == 0
        assert t["z_ref_vs_python"].value > t["z_enc_vs_z_ref"].value


def test_unknown_fixture_is_rejected():
    with pytest.raises(FixtureError, match="not declared"):
        load_json("no_such_fixture")
