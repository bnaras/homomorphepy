"""The Ct operator group and the Context/codec scheme workarounds.

These tests are the guard on decision D6: they assert that the two
upstream gaps stay covered. If openfhe-python later binds the missing
operators or GetSchemeId, these still pass — they test our surface, not
the absence of theirs.

Everything here needs a real backend, so the whole module skips when
openfhe is not importable.
"""

from __future__ import annotations

import functools
import operator

import pytest

from homomorphepy import (
    Ct,
    Scheme,
    fhe_context,
    have_backend,
    packed_codec,
    set_thread_env,
)

set_thread_env(2)

pytestmark = [
    pytest.mark.openfhe,
    pytest.mark.skipif(not have_backend(), reason="openfhe not installed"),
]

TOL = 1e-6  # CKKS at scaling_mod_size=50, depth 1: ample margin


@pytest.fixture(scope="module")
def ckks():
    return fhe_context(
        "CKKS", multiplicative_depth=1, scaling_mod_size=50, batch_size=8
    )


@pytest.fixture(scope="module")
def keys(ckks):
    return ckks.KeyGen()


def enc(ctx, keys, values):
    codec = packed_codec(ctx)
    return Ct(ctx.Encrypt(keys.publicKey, codec.encode(values)), ctx.cc)


def dec(ctx, keys, ct, length):
    codec = packed_codec(ctx)
    pt = ctx.Decrypt(ct.raw if isinstance(ct, Ct) else ct, keys.secretKey)
    return codec.decode(pt, length)


class TestOperatorGroup:
    """P12: openfhe-python binds only __add__; we supply the rest."""

    def test_bare_ciphertext_lacks_the_operators(self, ckks, keys):
        # Documents the upstream gap this wrapper exists for. If this
        # ever fails, openfhe-python has fixed P12 and the wrapper can
        # become a thin pass-through.
        raw = enc(ckks, keys, [1.0, 2.0]).raw
        # The bare expressions are the point: each must raise.
        with pytest.raises(TypeError):
            raw * 2.0  # noqa: B018
        with pytest.raises(TypeError):
            raw - raw  # noqa: B018
        with pytest.raises(TypeError):
            -raw  # noqa: B018

    def test_add(self, ckks, keys):
        a = enc(ckks, keys, [1.5, 2.5])
        got = dec(ckks, keys, a + a, 2)
        assert got == pytest.approx([3.0, 5.0], abs=TOL)

    def test_sub(self, ckks, keys):
        a, b = enc(ckks, keys, [5.0, 7.0]), enc(ckks, keys, [1.0, 2.0])
        assert dec(ckks, keys, a - b, 2) == pytest.approx([4.0, 5.0], abs=TOL)

    def test_mult_by_scalar(self, ckks, keys):
        a = enc(ckks, keys, [1.5, 2.0])
        assert dec(ckks, keys, a * 4.0, 2) == pytest.approx([6.0, 8.0], abs=TOL)

    def test_scalar_mult_is_commutative(self, ckks, keys):
        a = enc(ckks, keys, [1.5, 2.0])
        assert dec(ckks, keys, 4.0 * a, 2) == pytest.approx([6.0, 8.0], abs=TOL)

    def test_negate(self, ckks, keys):
        a = enc(ckks, keys, [1.5, -2.0])
        assert dec(ckks, keys, -a, 2) == pytest.approx([-1.5, 2.0], abs=TOL)

    def test_reversed_subtraction(self, ckks, keys):
        # OpenFHE has no reversed subtraction; computed as -(ct - s),
        # mirroring openfhe.R methods-eval.R:42-44.
        a = enc(ckks, keys, [1.0, 2.0])
        assert dec(ckks, keys, 10.0 - a, 2) == pytest.approx([9.0, 8.0], abs=TOL)

    def test_division_by_public_scalar(self, ckks, keys):
        a = enc(ckks, keys, [6.0, 8.0])
        assert dec(ckks, keys, a / 4.0, 2) == pytest.approx([1.5, 2.0], abs=TOL)

    def test_division_by_ciphertext_is_refused(self, ckks, keys):
        a = enc(ckks, keys, [6.0, 8.0])
        with pytest.raises(TypeError, match="no homomorphic reciprocal"):
            a / a


class TestAggregationIdioms:
    """The reason the wrapper exists: protocol code reads like R's."""

    def test_sum_over_sites(self, ckks, keys):
        # R writes Reduce(`+`, encrypted_locals) in master_aggregate;
        # sum() needs __radd__ because it starts from int 0.
        cts = [enc(ckks, keys, [float(i), 1.0]) for i in (1, 2, 3, 4)]
        assert dec(ckks, keys, sum(cts), 2) == pytest.approx([10.0, 4.0], abs=TOL)

    def test_reduce_operator_add(self, ckks, keys):
        cts = [enc(ckks, keys, [1.0, 2.0]) for _ in range(3)]
        total = functools.reduce(operator.add, cts)
        assert dec(ckks, keys, total, 2) == pytest.approx([3.0, 6.0], abs=TOL)

    def test_mean_across_sites(self, ckks, keys):
        # The consensus-ADMM idiom: ct_avg <- Reduce(`+`, cts) * (1/N)
        cts = [enc(ckks, keys, [float(v)]) for v in (2.0, 4.0, 6.0)]
        avg = functools.reduce(operator.add, cts) * (1.0 / len(cts))
        assert dec(ckks, keys, avg, 1) == pytest.approx([4.0], abs=TOL)


class TestCtSemantics:
    def test_attribute_passthrough(self, ckks, keys):
        a = enc(ckks, keys, [1.0])
        assert a.GetLevel() == a.raw.GetLevel()

    def test_double_wrapping_is_idempotent(self, ckks, keys):
        a = enc(ckks, keys, [1.0])
        assert Ct(a, ckks.cc).raw is a.raw

    def test_equality_is_refused(self, ckks, keys):
        a = enc(ckks, keys, [1.0])
        with pytest.raises(TypeError, match="cannot be compared"):
            a == a  # noqa: B015


class TestContextRemembersScheme:
    """P14: openfhe-python does not bind GetSchemeId."""

    def test_bare_context_cannot_report_its_scheme(self, ckks):
        # The gap the Context wrapper closes.
        assert not hasattr(ckks.cc, "GetSchemeId")

    def test_ckks_scheme_recorded(self, ckks):
        assert ckks.scheme is Scheme.CKKS
        assert ckks.scheme.is_approximate

    def test_bfv_scheme_recorded(self):
        ctx = fhe_context("BFV", plaintext_modulus=65537, multiplicative_depth=1)
        assert ctx.scheme is Scheme.BFV
        assert not ctx.scheme.is_approximate

    def test_params_are_recorded_for_drift_diffing(self, ckks):
        assert ckks.params["scaling_mod_size"] == 50
        assert ckks.params["multiplicative_depth"] == 1

    def test_unknown_parameter_is_rejected(self):
        # Silently ignoring a parameter would change the security level
        # or noise budget with no visible symptom.
        with pytest.raises(TypeError, match="unknown parameter"):
            fhe_context("CKKS", multiplicative_depth=1, scaling_mod_sixe=50)

    def test_default_features_match_R(self, ckks, keys):
        # fhe_context enables PKE|KEYSWITCH|LEVELEDSHE like the R
        # constructor; if it did not, this encrypt/add/decrypt fails.
        a = enc(ckks, keys, [1.0])
        assert dec(ckks, keys, a + a, 1) == pytest.approx([2.0], abs=TOL)


class TestCodecFollowsScheme:
    def test_ckks_codec_round_trips_reals(self, ckks, keys):
        got = dec(ckks, keys, enc(ckks, keys, [1.25, -0.5]), 2)
        assert got == pytest.approx([1.25, -0.5], abs=TOL)

    def test_bfv_codec_round_trips_exact_integers(self):
        ctx = fhe_context("BFV", plaintext_modulus=65537, multiplicative_depth=1)
        k = ctx.KeyGen()
        codec = packed_codec(ctx)
        ct = Ct(ctx.Encrypt(k.publicKey, codec.encode([3, 4, 5])), ctx.cc)
        total = ct + ct
        pt = ctx.Decrypt(total.raw, k.secretKey)
        # BFV is exact: assert equality, never a tolerance.
        assert codec.decode(pt, 3) == [6, 8, 10]

    def test_codec_refuses_a_bare_context(self, ckks):
        with pytest.raises(TypeError, match="upstream P14"):
            packed_codec(ckks.cc)
