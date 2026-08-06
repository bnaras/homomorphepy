"""Site/Master protocol behavior, against real ciphertexts.

Mirrors homomorpheR's tinytest coverage for the supported (non-Paillier)
actor surface: test_sites.R, test_master_worker.R, test_ckks_master.R,
test_threshold_master.R and test_threshold_bfv.R.

The threshold tests additionally check the property the R version
cannot have, because its master holds every share: that no secret
material sits on the master at all.
"""

from __future__ import annotations

import math

import pytest

from homomorphepy import (
    CKKSMaster,
    Site,
    ThresholdSite,
    fhe_context,
    have_backend,
    load_json,
    make_ckks_master,
    make_threshold_master,
    make_worker,
    set_thread_env,
)

set_thread_env(2)

pytestmark = [
    pytest.mark.openfhe,
    pytest.mark.skipif(not have_backend(), reason="openfhe not installed"),
]

TOL = 1e-6


def mean_fn(data, theta):
    """A trivial local summary: sum of (x - theta)."""
    return float(sum(x - theta for x in data))


@pytest.fixture(scope="module")
def ckks():
    return fhe_context(
        "CKKS", multiplicative_depth=1, scaling_mod_size=50, batch_size=8
    )


class TestSite:
    def test_summary(self):
        s = Site("A", [1.0, 2.0, 3.0], mean_fn)
        assert s.summary(0.0) == 6.0

    def test_none_signals_non_evaluable(self):
        s = Site("A", None, lambda d, t: None)
        assert s.summary(0.0) is None

    def test_nan_also_signals_non_evaluable(self):
        # R's local_fn returns NA; a Python local_fn is as likely to
        # produce NaN from a failed solve, so both are accepted.
        s = Site("A", None, lambda d, t: float("nan"))
        assert s.summary(0.0) is None


class TestCKKSMaster:
    def test_aggregate_matches_cleartext(self, ckks):
        sites = [
            make_worker("S1", [1.0, 2.0], mean_fn),
            make_worker("S2", [3.0, 4.0], mean_fn),
            make_worker("S3", [5.0], mean_fn),
        ]
        m = make_ckks_master("M", ckks, ckks.KeyGen()).set_workers(sites)
        expected = sum(mean_fn(s.data, 0.5) for s in sites)
        assert m.aggregate(0.5) == pytest.approx(expected, abs=TOL)

    def test_non_evaluable_site_yields_nan(self, ckks):
        good = make_worker("good", [1.0], mean_fn)
        bad = make_worker("bad", None, lambda d, t: None)
        m = make_ckks_master("M", ckks, ckks.KeyGen()).set_workers([good, bad])
        assert math.isnan(m.aggregate(0.0))

    def test_workers_receive_the_public_key(self, ckks):
        s = make_worker("S", [1.0], mean_fn)
        m = make_ckks_master("M", ckks, ckks.KeyGen()).set_workers([s])
        assert s.public_key is m.public_key

    def test_aggregate_without_workers_is_an_error(self, ckks):
        m = CKKSMaster("M", ckks, ckks.KeyGen())
        with pytest.raises(RuntimeError, match="no workers"):
            m.aggregate(0.0)

    def test_master_requires_a_wrapped_context(self, ckks):
        # A bare CryptoContext cannot report its scheme (P14).
        with pytest.raises(TypeError, match="homomorphepy Context"):
            CKKSMaster("M", ckks.cc, ckks.KeyGen())


class TestThresholdMaster:
    @pytest.fixture
    def threshold(self, ckks):
        sites = [
            ThresholdSite("S1", [1.0, 2.0], mean_fn),
            ThresholdSite("S2", [3.0, 4.0], mean_fn),
            ThresholdSite("S3", [5.0, 6.0], mean_fn),
        ]
        return make_threshold_master("M", ckks, sites), sites

    def test_aggregate_matches_cleartext(self, threshold):
        m, sites = threshold
        expected = sum(mean_fn(s.data, 1.0) for s in sites)
        assert m.aggregate(1.0) == pytest.approx(expected, abs=TOL)

    def test_master_holds_no_secret_material(self, threshold):
        # The property homomorpheR's master cannot have: it stores every
        # sk_i. Here the shares live at the sites, which is what makes a
        # cross-process (or cross-language) split possible without
        # shipping private keys.
        m, sites = threshold
        assert not hasattr(m, "secret_keys")
        assert not hasattr(m, "secret_share")
        assert all(s.secret_share is not None for s in sites)

    def test_every_site_holds_a_distinct_share(self, threshold):
        _, sites = threshold
        assert len({id(s.secret_share) for s in sites}) == len(sites)

    def test_exactly_one_lead(self, threshold):
        _, sites = threshold
        assert [s._is_lead for s in sites] == [True, False, False]

    def test_fusion_requires_the_lead_first(self, threshold):
        # OpenFHE requires the lead partial first; permuting yields
        # garbage rather than an error, so the master refuses.
        m, sites = threshold
        sites[0]._is_lead, sites[1]._is_lead = False, True
        try:
            with pytest.raises(RuntimeError, match="must be the lead"):
                m.aggregate(0.0)
        finally:
            sites[0]._is_lead, sites[1]._is_lead = True, False

    def test_two_sites_minimum(self, ckks):
        with pytest.raises(ValueError, match="at least two sites"):
            make_threshold_master("M", ckks, [ThresholdSite("only", [1.0], mean_fn)])

    def test_plain_sites_are_rejected(self, ckks):
        with pytest.raises(TypeError, match="ThresholdSite"):
            make_threshold_master(
                "M", ckks, [Site("a", [1.0], mean_fn), Site("b", [2.0], mean_fn)]
            )

    def test_non_evaluable_site_yields_nan(self, ckks):
        sites = [
            ThresholdSite("ok", [1.0], mean_fn),
            ThresholdSite("bad", None, lambda d, t: None),
        ]
        m = make_threshold_master("M", ckks, sites)
        assert math.isnan(m.aggregate(0.0))


class TestThresholdBFVExactCounting:
    """The query-count-threshold example's core, under exact BFV."""

    def test_counts_are_exact_across_sites(self):
        ctx = fhe_context("BFV", plaintext_modulus=65537, multiplicative_depth=1)
        f = load_json("query_count")

        def count_fn(rows, _theta):
            return sum(
                1
                for i in range(len(rows["age"]))
                if rows["age"][i] < 50 and rows["sex"][i] == "F" and rows["bm"][i] < 0.2
            )

        sites = [
            ThresholdSite(f"site{i + 1}", rows, count_fn)
            for i, rows in enumerate(f["sites"])
        ]
        m = make_threshold_master("M", ctx, sites)

        got = m.aggregate(None)
        # BFV is exact arithmetic: assert equality against the value R
        # computed, not a tolerance.
        assert got == f["expected"]["total"]
        assert got == sum(f["expected"]["per_site"])
