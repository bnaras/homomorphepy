"""Adversarial tests: the failure modes, not the happy path.

Every assertion here corresponds to something that used to succeed
silently and return a wrong number, or that used to look configured
while it was not. The arithmetic tests elsewhere pass either way, which
is exactly why these are separate: a protocol bug that does not change
the answer on a well-behaved run is invisible to them.

Two of these were measured before they were fixed, on this backend:
decrypting a BFV ciphertext under the wrong key returned 17809 for a
true 7, and two BFV values of 40000 summed to 14463 under t = 65537.
Neither raised anything.
"""

from __future__ import annotations

import math

import pytest

from homomorphepy import (
    BadContribution,
    KeyMismatch,
    OpenFHEParams,
    PublicParams,
    RemoteSite,
    Site,
    SiteUnavailable,
    backend,
    fhe_context,
    have_backend,
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


def nll(data, lam):
    """A Poisson negative log-likelihood, as a stand-in local summary."""
    return -sum(x * math.log(lam) - lam - math.lgamma(x + 1) for x in data)


@pytest.fixture(scope="module")
def cc():
    return fhe_context(
        "CKKS", multiplicative_depth=1, scaling_mod_size=50, batch_size=8
    )


@pytest.fixture(scope="module")
def ccm():
    return fhe_context(
        "CKKS",
        multiplicative_depth=1,
        scaling_mod_size=50,
        batch_size=8,
        features=[backend().PKESchemeFeature.MULTIPARTY],
    )


@pytest.fixture(scope="module")
def ccb():
    return fhe_context(
        "BFV",
        multiplicative_depth=1,
        plaintext_modulus=65537,
        features=[backend().PKESchemeFeature.MULTIPARTY],
    )


class _Far(RemoteSite):
    """A remote proxy with nothing implemented: everything must refuse."""

    def __init__(self, name):
        super().__init__(name, data=None, local_fn=None)

    def contribute(self, theta):
        raise NotImplementedError("no transport")


class _Near(RemoteSite):
    """A remote proxy that provisions its endpoint and returns ciphertext."""

    def __init__(self, name, rows):
        super().__init__(name, data=rows, local_fn=None)

    def set_public_params(self, params):
        self._params = params
        return self

    def contribute(self, theta):
        return self.params.encrypt(nll(self.data, theta))


class _Leaky(_Near):
    """A remote proxy that replies in cleartext — the silent one."""

    def contribute(self, theta):
        return nll(self.data, theta)


class _Unreachable(_Near):
    """A remote proxy whose transport is down."""

    def contribute(self, theta):
        raise SiteUnavailable("connection refused")

    def keygen_round(self, ctx, prev_pk=None):
        return Site.keygen_round(self, ctx, prev_pk)

    def partial_decrypt(self, ciphertext, lead=False):
        raise SiteUnavailable("connection refused")


class TestUnreachableIsNotNonEvaluable:
    """Returning None and being unreachable are different events.

    None means *this theta broke my solver*, and the optimizer should
    back off. Unreachable means the round cannot be completed, and
    carrying on would sum over a different set of sites -- silently
    changing the objective between optimizer iterations.
    """

    def test_an_unreachable_worker_aborts_the_round(self, cc):
        m = make_ckks_master("U", cc, cc.KeyGen())
        m.set_workers([make_worker("S1", [2, 3], nll), _Unreachable("Down", [6, 7])])
        with pytest.raises(SiteUnavailable) as exc:
            m.aggregate(3.5)
        # The name travels, not the site: a site drags its data and its
        # key share into anything that logs the exception.
        assert exc.value.site_name == "Down"

    def test_an_unreachable_site_loses_the_whole_threshold_round(self, ccm):
        # n-of-n: one missing partial costs the entire result, not one
        # summand.
        good = make_worker("Up", [1.0], nll)
        down = _Unreachable("Down", [2.0])
        m = make_threshold_master("UM", ccm, [good, down])
        with pytest.raises(SiteUnavailable, match="n-of-n"):
            m.decrypt(good.encrypt(1.0))


class TestPublicParamsCarryNoSecret:
    def test_bundle_has_only_context_and_public_key(self, cc):
        m = make_ckks_master("M", cc, cc.KeyGen())
        w1, w2 = make_worker("S1", [2, 3], nll), make_worker("S2", [4, 5], nll)
        m.set_workers([w1, w2])

        p = w1.params
        assert isinstance(p, PublicParams)
        assert set(OpenFHEParams.__slots__) == {"_ctx", "_pk"}
        assert not any(k in repr(p).lower() for k in ("sk", "secret key", "private"))
        assert "secret material: none" in repr(p)

    def test_single_decrypter_master_keeps_its_key_in_one_place(self, cc):
        m = make_ckks_master("M", cc, cc.KeyGen())
        m.set_workers([make_worker("S1", [2, 3], nll)])
        # The secret half lives in the keypair and nowhere else -- not
        # duplicated loose on the object, where it would be a second
        # thing to reason about and nothing would read it.
        assert set(vars(m)) == {"name", "ctx", "workers", "keypair"}


class TestSiteRewiring:
    def test_a_site_cannot_be_silently_rewired_to_a_second_master(self, cc):
        w = make_worker("S1", [2, 3], nll)
        make_ckks_master("A", cc, cc.KeyGen()).set_workers([w])
        other = make_ckks_master("B", cc, cc.KeyGen())
        with pytest.raises(ValueError, match="already holds different public"):
            other.set_workers([w])

    def test_rewiring_to_the_same_master_is_allowed(self, cc):
        w = make_worker("S1", [2, 3], nll)
        m = make_ckks_master("A", cc, cc.KeyGen())
        m.set_workers([w])
        m.set_workers([w])  # the same setup message twice

    def test_an_unconfigured_site_says_so_rather_than_encrypting(self):
        loose = make_worker("Loose", [1, 2], nll)
        with pytest.raises(RuntimeError, match="no public parameters"):
            _ = loose.params
        with pytest.raises(RuntimeError, match="no public parameters"):
            loose.contribute(1.0)


class TestRemoteSetupFailsClosed:
    def test_the_base_class_refuses_all_three_site_side_steps(self, cc, ccm):
        m = make_ckks_master("M", cc, cc.KeyGen())
        w = make_worker("S", [1.0], nll)
        m.set_workers([w])
        p = w.params
        far = _Far("Far")

        with pytest.raises(NotImplementedError, match="set_public_params"):
            far.set_public_params(p)
        with pytest.raises(NotImplementedError, match="keygen_round"):
            far.keygen_round(ccm)
        with pytest.raises(NotImplementedError, match="partial_decrypt"):
            far.partial_decrypt(p.encrypt(1.0))

    def test_wiring_an_unprovisioned_proxy_leaves_the_master_unwired(self, cc):
        m = make_ckks_master("F", cc, cc.KeyGen())
        with pytest.raises(NotImplementedError, match="set_public_params"):
            m.set_workers([_Far("Far")])
        # Unwired, not half-wired.
        assert m.workers == []

    def test_a_subclass_that_provisions_takes_part_unchanged(self, cc):
        m = make_ckks_master("N", cc, cc.KeyGen())
        m.set_workers([make_worker("S1", [2, 3], nll), _Near("Far", [6, 7])])
        assert m.aggregate(3.5) == pytest.approx(nll([2, 3, 6, 7], 3.5), abs=1e-3)


class TestCleartextReplyNeverReachesTheTotal:
    def test_a_cleartext_reply_is_refused(self, cc):
        # This is the one that returned the *right* answer. A site
        # handed the aggregator its individual contribution in the
        # clear, ordinary addition folded it in, and nothing noticed.
        m = make_ckks_master("L", cc, cc.KeyGen())
        m.set_workers([make_worker("S1", [2, 3], nll), _Leaky("Leak", [6, 7])])
        with pytest.raises(BadContribution, match="Leak"):
            m.aggregate(3.5)


class TestForeignCiphertexts:
    def test_a_ciphertext_from_another_key_is_refused(self, cc):
        keys, keys_b = cc.KeyGen(), cc.KeyGen()
        m = make_ckks_master("M", cc, keys)
        m.set_workers([make_worker("S1", [2, 3], nll)])
        mine = OpenFHEParams(cc, keys.publicKey)
        theirs = OpenFHEParams(cc, keys_b.publicKey)

        with pytest.raises(KeyMismatch):
            m.decrypt(theirs.encrypt(1.0))
        assert m.decrypt(mine.encrypt(1.0)) == pytest.approx(1.0, abs=1e-6)

    def test_a_site_refuses_to_apply_its_share_to_a_foreign_ciphertext(self, ccm):
        # The site checks for itself, with the joint key it was given
        # at setup. It asks no one for anything.
        o1, o2 = make_worker("O1", [1], nll), make_worker("O2", [2], nll)
        make_threshold_master("OM", ccm, [o1, o2])
        t1, t2 = make_worker("T1", [1], nll), make_worker("T2", [2], nll)
        make_threshold_master("TM", ccm, [t1, t2])

        with pytest.raises(KeyMismatch):
            o1.partial_decrypt(t1.encrypt(1.0), lead=True)
        o1.partial_decrypt(o1.encrypt(1.0), lead=True)  # its own: fine


class TestMasterConstruction:
    def test_make_ckks_master_requires_a_ckks_context(self):
        bfv = fhe_context("BFV", multiplicative_depth=1, plaintext_modulus=65537)
        with pytest.raises(ValueError, match="CKKS"):
            make_ckks_master("X", bfv, bfv.KeyGen())

    def test_a_threshold_master_refuses_to_be_rewired(self, ccm):
        s1, s2 = make_worker("A", [1], nll), make_worker("B", [2], nll)
        m = make_threshold_master("M", ccm, [s1, s2])
        with pytest.raises(RuntimeError, match="already wired"):
            m.set_workers([s1, s2])


class TestThresholdSetupGuards:
    def test_the_same_party_listed_twice_is_rejected(self, ccm):
        d1, d2 = make_worker("D1", [1], nll), make_worker("D2", [2], nll)
        with pytest.raises(ValueError, match="the same party"):
            make_threshold_master("D", ccm, [d1, d2, d1])
        # ... and the ceremony left no trace on the sites it visited.
        assert not d1.has_share
        assert not d2.has_share

    def test_duplicate_names_are_rejected(self, ccm):
        with pytest.raises(ValueError, match="share the name"):
            make_threshold_master(
                "D", ccm, [make_worker("A", [1], nll), make_worker("A", [2], nll)]
            )

    def test_a_non_site_is_rejected(self, ccm):
        with pytest.raises(TypeError, match="not a Site"):
            make_threshold_master("D", ccm, [make_worker("A", [1], nll), 42])

    def test_a_site_already_in_a_protocol_cannot_join_a_second(self, ccm):
        t1, t2 = make_worker("T1", [2, 3], nll), make_worker("T2", [4, 5], nll)
        make_threshold_master("TM", ccm, [t1, t2])
        with pytest.raises(ValueError, match="already taking part"):
            make_threshold_master("TM2", ccm, [t1, t2])

    def test_a_ceremony_that_fails_partway_rolls_back(self, ccm):
        # Otherwise the sites already visited would hold a share
        # belonging to a ceremony that never completed, and the check
        # above would then refuse them a retry.
        r1, r2 = make_worker("R1", [1], nll), make_worker("R2", [2], nll)
        with pytest.raises(NotImplementedError, match="keygen_round"):
            make_threshold_master("R", ccm, [r1, r2, _Far("RFar")])
        assert not r1.has_share
        assert r1._ctx is None
        assert not r2.has_share
        # The same sites work on a retry.
        make_threshold_master("R2", ccm, [r1, r2])


class TestExactSchemesRefuseWhatTheyCannotCarry:
    """BFV/BGV previously coerced with int(): 0.9 became 0."""

    @staticmethod
    def _pair(f, ctx):
        return make_threshold_master(
            "B", ctx, [make_worker("X1", None, f), make_worker("X2", None, f)]
        )

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (0.9, "not an integer"),
            (float("inf"), "NaN or an infinity"),
            (40000, "plaintext modulus"),
        ],
    )
    def test_refused(self, ccb, value, match):
        m = self._pair(lambda d, t: value, ccb)
        with pytest.raises(ValueError, match=match):
            m.aggregate(1)

    def test_integers_still_go_through_exactly(self, ccb):
        assert self._pair(lambda d, t: 5, ccb).aggregate(1) == 10

    def test_bgv_refuses_a_non_integer_too(self):
        bgv = fhe_context(
            "BGV",
            multiplicative_depth=1,
            plaintext_modulus=65537,
            features=[backend().PKESchemeFeature.MULTIPARTY],
        )
        with pytest.raises(ValueError, match="not an integer"):
            self._pair(lambda d, t: 2.7, bgv).aggregate(1)

    def test_non_evaluable_still_reaches_the_caller(self, ccb):
        # None is the documented plaintext reply, and it must reach the
        # caller rather than the codec.
        m = make_threshold_master(
            "NM",
            ccb,
            [
                make_worker("N1", None, lambda d, t: None),
                make_worker("N2", None, lambda d, t: 1),
            ],
        )
        assert math.isnan(m.aggregate(1))


class TestValidators:
    @pytest.mark.parametrize("bad", ["", "   ", None, ["a", "b"], 7])
    def test_a_party_needs_a_usable_name(self, bad):
        with pytest.raises(ValueError, match="non-empty name"):
            make_worker(bad, [1], nll)

    def test_a_site_needs_a_callable_local_fn(self):
        with pytest.raises(TypeError, match="callable local_fn"):
            make_worker("x", [1], 42)

    def test_only_a_publicparams_configures_a_site(self):
        with pytest.raises(TypeError, match="PublicParams"):
            make_worker("z", [1], nll).set_public_params({"pk": 1})
