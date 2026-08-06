"""Cox-lasso on DLBCL: the encrypted pipeline, against R's results.

Real data, so unlike the simulated examples there is no draw to vary
and the fixture *is* the data. Comparisons are therefore against R's
shipped values.

The expensive ADMM loop (~150 iterations x 3 conic solves at K=100) is
exercised in one test; the standardization and screening stages, which
are where the encrypted channel does the most interesting work, run
cheaply and are tested separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from homomorphepy import have_backend, load_golden, set_thread_env

set_thread_env(2)

pytest.importorskip("cvxpy", reason="install the 'stats' extra")

pytestmark = [
    pytest.mark.openfhe,
    pytest.mark.skipif(not have_backend(), reason="openfhe not installed"),
]


@pytest.fixture(scope="module")
def screened():
    """Standardization + screening only. Cheap enough for many tests."""
    from homomorphepy.examples import cox_lasso

    return cox_lasso.run(recompute_admm=False)


class TestEncryptedStandardization:
    def test_pooled_moments_match_cleartext(self, screened):
        # Column means and SDs pooled across sites under encryption.
        # CKKS error at essentially float64 precision.
        assert screened.pool_agree_mu < 1e-12
        assert screened.pool_agree_sigma < 1e-12

    def test_matches_the_magnitude_R_reports(self, screened):
        # R's shipped pool_agree is ~4.4e-16 (mu) and ~3.1e-15 (sigma);
        # ours should be the same order, not merely "small".
        g = load_golden()
        assert screened.pool_agree_mu == pytest.approx(g["pool_agree"]["mu"], abs=1e-14)
        assert screened.pool_agree_sigma == pytest.approx(
            g["pool_agree"]["sigma"], abs=1e-13
        )


class TestEncryptedScreening:
    """The stage the cryptography review flagged as knife-edged."""

    def test_selects_the_same_probes_as_R(self, screened):
        golden = np.asarray(load_golden()["top_idx"], dtype=int)
        # 6416 probes ranked by a univariate Cox score pooled under
        # encryption, top 100 kept. Near-ties at the K=100 boundary
        # were predicted to flip under a one-ulp perturbation. They do
        # not -- but only because the expression matrix ships as raw
        # float64 rather than CSV, and because both sort traps are
        # handled: lexsort key order, and a stable argsort.
        assert set(screened.top_idx.tolist()) == set(golden.tolist())
        assert len(set(screened.top_idx.tolist())) == 100

    def test_rank_order_also_matches(self, screened):
        # Stronger than set equality: the probes come out in the same
        # order, so the |Z| statistics agree to better than the gaps
        # between adjacent ranks.
        golden = np.asarray(load_golden()["top_idx"], dtype=int)
        assert np.array_equal(screened.top_idx, golden)

    def test_indices_are_one_based_like_R(self, screened):
        assert screened.top_idx.min() >= 1
        assert screened.top_idx.max() <= 6416


@pytest.mark.slow
class TestFullPipeline:
    """The ADMM loop. ~27 minutes, so opt-in: `pytest -m slow`.

    Measured 2026-08-06 against R's shipped results:

        iterations                        python 147, R 147
        max |python_admm - R z_enc|       1.556e-07
        max |python_admm - R z_ref|       1.437e-07
        max |python_central - R agg_beta| 1.991e-07
        non-zero coefficients             python 38, R 38
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def full():
        from homomorphepy.examples import cox_lasso

        return cox_lasso.run(recompute_admm=True)

    def test_admm_converges(self, full):
        from homomorphepy.examples.cox_lasso import MAX_ITER

        assert full.n_iter is not None and full.n_iter < MAX_ITER

    def test_iteration_count_is_comparable_to_R(self, full):
        r_n_iter = int(load_golden()["n_iter_enc"])
        # Deliberately NOT an equality, even though it MATCHED exactly
        # (147 = 147) when measured. The stopping rule is an absolute
        # residual threshold, so the count is where a continuous
        # quantity first crosses it, and test_consensus_admm shows that
        # flipping under CKKS noise alone at n=23. This problem simply
        # is not near a boundary -- consistent with R reproducing 147
        # across an openfhe.R version change. Asserting equality would
        # convert a happy fact into a brittle requirement.
        assert abs(full.n_iter - r_n_iter) < 20

    def test_coefficients_match_Rs_encrypted_fit(self, full):
        # Both are ADMM iterates stopped at the same residual
        # tolerance (5e-3) rather than exact optima, so agreement was
        # expected only at convergence scale. Measured 1.6e-07 -- four
        # orders tighter, because the trajectories track each other
        # rather than merely landing in the same basin.
        r_z_enc = np.asarray(load_golden()["z_enc"], dtype=float)
        assert np.max(np.abs(full.beta_admm - r_z_enc)) < 1e-5

    def test_centralized_fit_matches_Rs(self, full):
        # No ADMM involved: two independent Clarabel builds on the
        # same convex problem, reached through two modelling layers.
        # Measured 2.0e-07.
        r_agg = np.asarray(load_golden()["agg_beta"], dtype=float)
        assert np.max(np.abs(full.beta_centralized - r_agg)) < 1e-5

    def test_selects_a_sparse_model(self, full):
        # The lasso is doing something: K=100 screened probes in,
        # substantially fewer retained.
        nz = full.n_nonzero
        assert 0 < nz < 100
        # Same count as R (38), so the penalty selects identically.
        r_z_enc = np.asarray(load_golden()["z_enc"], dtype=float)
        assert nz == int(np.sum(np.abs(r_z_enc) > 1e-8))

    def test_active_set_agrees_with_R(self, full):
        # Which probes survive the penalty matters more than their
        # exact values; a differing active set would be a real
        # divergence rather than a tolerance question.
        r_z_enc = np.asarray(load_golden()["z_enc"], dtype=float)
        py = {i for i, v in enumerate(full.beta_admm) if abs(v) > 1e-6}
        r = {i for i, v in enumerate(r_z_enc) if abs(v) > 1e-6}
        jaccard = len(py & r) / max(len(py | r), 1)
        assert jaccard > 0.9, f"active sets diverge: {jaccard:.2f}"
