"""The worked examples, asserted per the plan's tolerance ladder.

Three rungs, chosen per example rather than applied uniformly:

* **exact** — BFV integer results, compared with ``==``. No tolerance
  is admissible: a count that is off by anything is wrong.
* **CKKS tolerance** — encrypted vs plaintext. The real cryptographic
  claim, and the only place approximation is expected.
* **statistical** — a fitted estimate against a cleartext reference,
  where optimizer behavior dominates CKKS noise by orders of magnitude.

Conflating the last two is how a solver difference gets misreported as
crypto noise, so each assertion states which rung it is on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from homomorphepy import have_backend, set_thread_env

set_thread_env(2)

pytestmark = [
    pytest.mark.openfhe,
    pytest.mark.skipif(not have_backend(), reason="openfhe not installed"),
]

CKKS_TOL = 1e-6  # depth 1-2 at scaling_mod_size 50
STATISTICAL_TOL = 1e-4  # optimizer-scale, not crypto-scale


class TestAggregation:
    @pytest.fixture(scope="class")
    @staticmethod
    def result():
        from homomorphepy.examples import aggregation

        return aggregation.run()

    def test_encrypted_total_is_exact(self, result):
        # BFV: assert equality, never a tolerance.
        assert result.exact
        assert result.total_encrypted == result.total_cleartext

    def test_total_is_the_sum_of_site_counts(self, result):
        assert result.total_encrypted == sum(result.per_site_cleartext)

    def test_runs_over_the_full_cohorts(self, result):
        assert result.site_sizes == [1000, 500, 1500]


class TestQueryCount:
    @pytest.fixture(scope="class")
    @staticmethod
    def result():
        from homomorphepy.examples import query_count

        return query_count.run()

    def test_encrypted_total_is_exact(self, result):
        # BFV is exact integer arithmetic, so the encrypted total must
        # equal the pooled cleartext total with no tolerance. The
        # cohort is simulated per run, so this is an internal-
        # consistency claim: the protocol reproduces the right answer
        # on whatever data it was given.
        assert result.exact
        assert result.total_encrypted == result.total_cleartext

    def test_total_is_the_sum_of_site_counts(self, result):
        assert result.total_encrypted == sum(result.per_site_cleartext)

    def test_runs_over_the_simulated_cohorts(self, result):
        assert result.site_sizes == [60, 15, 25]

    def test_no_party_can_decrypt_alone(self, result):
        # The distinction from the aggregation example: shares live at
        # the sites and the master holds none.
        assert len(result.master.sites) == 3
        assert not hasattr(result.master, "secret_share")
        assert all(s.secret_share is not None for s in result.master.sites)

    def test_holds_across_draws(self, result):
        # The claim is about the protocol, not one dataset: a different
        # draw must still recover its own cleartext total exactly.
        from homomorphepy.examples import query_count

        other = query_count.run(seed=7)
        assert other.exact
        assert other.total_encrypted != 0


class TestMLE:
    @pytest.fixture(scope="class")
    @staticmethod
    def result():
        from homomorphepy.examples import mle

        return mle.run()

    def test_converges(self, result):
        assert result.converged

    def test_matches_the_pooled_cleartext_fit(self, result):
        # Statistical rung. The Poisson MLE is the sample mean in closed
        # form, so the reference is exact and only the optimizer path
        # (through encryption) is under test.
        assert result.abs_difference < STATISTICAL_TOL

    def test_uses_all_the_data(self, result):
        from homomorphepy.examples import mle

        pooled = np.concatenate(mle.simulate())
        assert result.n_total == 40
        assert result.site_sizes == [20, 7, 13]
        assert result.lambda_pooled == pytest.approx(float(pooled.mean()), abs=1e-12)

    def test_the_optimizer_really_ran_through_the_channel(self, result):
        # Each objective call is a full encrypt/sum/decrypt round; a
        # handful of calls would mean the fit short-circuited.
        assert result.n_objective_calls > 5

    def test_standard_error_is_analytic_not_quasi_newton(self, result):
        # SE from the Fisher information n/lambda, not fit.hess_inv --
        # the BFGS inverse-Hessian approximation is not a variance
        # estimate.
        assert result.std_error == pytest.approx(
            math.sqrt(result.lambda_encrypted / result.n_total), rel=1e-12
        )


class TestFiniteDifferenceStep:
    """Records what the CKKS noise floor actually costs the optimizer.

    A review predicted SciPy's default finite-difference step (~1.5e-8,
    versus R's ndeps = 1e-3) would fall below the CKKS noise floor and
    leave BFGS differentiating noise. Measured, it does not: at depth 1
    and scaling_mod_size 50 the decrypted objective carries ~1e-13
    absolute error on a value of ~100, so the implied gradient noise at
    the default step is ~1.7e-06 against a true gradient of ~0.78.

    These tests pin the measurement rather than the prediction. If a
    deeper circuit or a larger scaling factor ever changes the balance,
    they fail and say so.
    """

    def test_encrypted_objective_is_far_more_precise_than_predicted(self):
        from homomorphepy.actors import make_ckks_master, make_worker
        from homomorphepy.context import fhe_context
        from homomorphepy.examples import mle as mle_mod

        sites = mle_mod.simulate()
        ctx = fhe_context(
            "CKKS", multiplicative_depth=1, scaling_mod_size=50, batch_size=8
        )
        master = make_ckks_master("M", ctx, ctx.KeyGen()).set_workers(
            [make_worker(f"S{i}", d, mle_mod.local_nll) for i, d in enumerate(sites)]
        )
        objective = mle_mod.make_objective(master)

        lam = 9.0
        clear = sum(mle_mod.local_nll(d, lam) for d in sites)
        errors = [abs(objective([lam]) - clear) for _ in range(8)]

        # Magnitude ~100. A loose bound on purpose: the cohort is drawn
        # per run, so pinning the exact value would make this a test of
        # the random draw rather than of the objective's precision.
        assert 50.0 < clear < 200.0
        # Essentially float64 precision, not the ~1e-5 the prediction
        # would have required to disturb the difference quotient.
        assert max(errors) < 1e-10

    def test_both_step_sizes_converge(self):
        from scipy.optimize import minimize

        from homomorphepy.actors import make_ckks_master, make_worker
        from homomorphepy.context import fhe_context
        from homomorphepy.examples import mle as mle_mod

        sites = mle_mod.simulate()
        ctx = fhe_context(
            "CKKS", multiplicative_depth=1, scaling_mod_size=50, batch_size=8
        )
        master = make_ckks_master("M", ctx, ctx.KeyGen()).set_workers(
            [make_worker(f"S{i}", d, mle_mod.local_nll) for i, d in enumerate(sites)]
        )
        objective = mle_mod.make_objective(master)
        pooled = float(np.concatenate(sites).mean())

        default_err = abs(
            float(minimize(objective, x0=[5.0], method="BFGS").x[0]) - pooled
        )
        widened_err = abs(
            float(
                minimize(
                    objective,
                    x0=[5.0],
                    method="BFGS",
                    options={"finite_diff_rel_step": mle_mod.FINITE_DIFF_STEP},
                ).x[0]
            )
            - pooled
        )

        # Both land on the answer. The claim under test is that the
        # default is USABLE, contradicting the prediction that it would
        # stall or wander.
        assert default_err < STATISTICAL_TOL, (
            f"SciPy's default step erred by {default_err:.3g}; if this "
            f"now exceeds the statistical tolerance, the CKKS noise "
            f"floor has risen and FINITE_DIFF_STEP became load-bearing"
        )
        assert widened_err < STATISTICAL_TOL


class TestSecureInference:
    @pytest.fixture(scope="class")
    @staticmethod
    def result():
        from homomorphepy.examples import secure_inference

        return secure_inference.run()

    def test_encrypted_scores_match_cleartext(self, result):
        # CKKS rung, within-language: the actual cryptographic claim.
        assert result.max_error < CKKS_TOL
        assert result.scores_encrypted == pytest.approx(
            result.scores_cleartext, abs=CKKS_TOL
        )

    def test_all_patients_scored_in_one_ciphertext(self, result):
        # SIMD packing: eight patients, four multiply-adds.
        assert result.n_patients == 8
        assert len(result.scores_encrypted) == 8

    def test_risk_bands_agree_with_cleartext(self, result):
        from homomorphepy.examples.secure_inference import _risk_band

        assert result.bands == [_risk_band(s) for s in result.scores_cleartext]

    def test_model_is_extractable_despite_encryption(self):
        # The protocol protects the biomarkers and the coefficients in
        # transit; it does not protect the model from query access.
        from homomorphepy.examples.secure_inference import extract_model

        a = extract_model()
        # Through CKKS, so to CKKS precision rather than exactly.
        assert a["n_queries"] == 5
        assert a["max_weight_error"] < 1e-6
        assert a["bias_error"] < 1e-6
