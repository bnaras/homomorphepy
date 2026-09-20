"""Site/Master protocol behavior, against real ciphertexts.

Covers the supported (non-Paillier) actor surface: sites, the two
masters, the setup seam and the threshold ceremony. The failure modes
-- cleartext replies, foreign keys, repeated parties, exact-scheme
overflow -- live in ``test_adversarial.py``, because they are exactly
the ones an arithmetic test passes either way.
"""

from __future__ import annotations

import math

import pytest

from homomorphepy import (
    CKKSMaster,
    OpenFHEParams,
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
    def test_a_configured_site_returns_ciphertext(self, ckks):
        s = make_worker("A", [1.0, 2.0, 3.0], mean_fn)
        make_ckks_master("M", ckks, ckks.KeyGen()).set_workers([s])
        ct = s.contribute(0.0)
        # What leaves is encrypted, and stamped with the key it was
        # encrypted under -- which is what check_encrypted tests.
        assert ct.raw.GetKeyTag() == s.params.tag

    def test_none_signals_non_evaluable_before_encryption(self):
        # No params needed: a site that cannot evaluate says so without
        # reaching the codec.
        s = make_worker("A", None, lambda d, t: None)
        assert s.contribute(0.0) is None

    def test_nan_also_signals_non_evaluable(self):
        s = make_worker("A", None, lambda d, t: float("nan"))
        assert s.contribute(0.0) is None

    def test_a_site_holds_its_params_and_nothing_of_the_master(self, ckks):
        s = make_worker("S", [1.0], mean_fn)
        m = make_ckks_master("M", ckks, ckks.KeyGen())
        m.set_workers([s])
        assert isinstance(s.params, OpenFHEParams)
        # The site keeps no handle on whoever wired it.
        assert not any(v is m for v in vars(s).values())


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

    def test_workers_receive_the_setup_message(self, ckks):
        s = make_worker("S", [1.0], mean_fn)
        m = make_ckks_master("M", ckks, ckks.KeyGen()).set_workers([s])
        assert s.params.tag == m.keypair.publicKey.GetKeyTag()

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
            make_worker("S1", [1.0, 2.0], mean_fn),
            make_worker("S2", [3.0, 4.0], mean_fn),
            make_worker("S3", [5.0, 6.0], mean_fn),
        ]
        return make_threshold_master("M", ckks, sites), sites

    def test_aggregate_matches_cleartext(self, threshold):
        m, sites = threshold
        expected = sum(mean_fn(s.data, 1.0) for s in sites)
        assert m.aggregate(1.0) == pytest.approx(expected, abs=TOL)

    def test_master_holds_no_secret_material(self, threshold):
        # Structural, not a promise in prose: the master has no
        # attribute a share could occupy, and every share is at the
        # site that generated it. That is what makes a cross-process
        # (or cross-language) split possible without shipping private
        # keys.
        m, sites = threshold
        assert set(vars(m)) == {"name", "ctx", "workers", "joint_public_key"}
        assert all(s.has_share for s in sites)

    def test_every_site_holds_a_distinct_share(self, threshold):
        _, sites = threshold
        assert len({id(s.secret_share) for s in sites}) == len(sites)

    def test_the_lead_role_comes_from_position_not_from_the_site(self, threshold):
        # The master tells each site which role it is playing, fixed by
        # its position in the key-generation chain. A site does not
        # carry the role, so there is no flag to permute.
        _, sites = threshold
        assert not any(hasattr(s, "_is_lead") for s in sites)

    def test_every_site_shares_the_joint_key(self, threshold):
        m, sites = threshold
        joint = m.joint_public_key.GetKeyTag()
        assert all(s.params.tag == joint for s in sites)

    def test_two_sites_minimum(self, ckks):
        with pytest.raises(ValueError, match="at least two sites"):
            make_threshold_master("M", ckks, [make_worker("only", [1.0], mean_fn)])

    def test_non_evaluable_site_yields_nan(self, ckks):
        sites = [
            make_worker("ok", [1.0], mean_fn),
            make_worker("bad", None, lambda d, t: None),
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
            make_worker(f"site{i + 1}", rows, count_fn)
            for i, rows in enumerate(f["sites"])
        ]
        m = make_threshold_master("M", ctx, sites)

        got = m.aggregate(None)
        # BFV is exact arithmetic: assert equality against the value R
        # computed, not a tolerance.
        assert got == f["expected"]["total"]
        assert got == sum(f["expected"]["per_site"])
