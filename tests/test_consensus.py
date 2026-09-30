"""Consensus ADMM, with and without output DP.

Three kinds of check:

- **Cross-language, on R's exact draws** (``admm_dp_cohort`` and
  ``admm_cohort`` fixtures). The example simulates its own data with
  numpy; these fixtures hold the cohorts R drew, so the two languages
  can be compared value for value.
- **On a fresh draw**, the claims the example makes: the surrogate step
  picks a converging rho, and with the noise off the fixed-T protocol
  reaches the centralized fit.
- **Solver discipline**: the solver is named, never inherited from an
  automatic choice.

The noisy runs are not compared across languages or across runs: their
noise is drawn afresh, so only their behavior is tested.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from homomorphepy import have_backend, set_thread_env

set_thread_env(2)

pytest.importorskip("cvxpy", reason="install the 'stats' extra")

from homomorphepy.examples import dp  # noqa: E402
from homomorphepy.examples._consensus import (  # noqa: E402
    SOLVER,
    SUPPORTED_SOLVERS,
    admm_loop,
    centralized,
    plain_consensus,
)

pytestmark = [
    pytest.mark.openfhe,
    pytest.mark.skipif(not have_backend(), reason="openfhe not installed"),
]

# Encrypted vs plaintext on identical data: the cryptographic claim.
CKKS_TOL = 1e-5
# ADMM iterate vs the centralized optimum: stopping-tolerance scale
# (the loop stops at residuals below 1e-3), not crypto scale.
STATISTICAL_TOL = 1e-4


def _sites(block, p):
    return [
        (
            np.asarray(s["X"], dtype=float).reshape(s["n"], p),
            np.asarray(s["y"], dtype=float),
        )
        for s in block
    ]


@pytest.fixture(scope="module")
def r_dp():
    """R's cohort and surrogate from cvxr-consensus-admm-dp, and R's results."""
    from homomorphepy import load_json

    f = load_json("admm_dp_cohort")
    return {
        "cohort": _sites(f["cohort"], f["p"]),
        "surrogate": _sites(f["surrogate"], f["p"]),
        "ref": f["reference"],
        "lam": f["lambda"],
        "tol": f["tol"],
    }


class TestAgreesWithR:
    """Cross-language, on R's exact draws."""

    def test_design_matches_R(self, r_dp):
        d = dp.CONSENSUS_DGP
        assert tuple(r_dp["ref"]["rho_sweep"]["rho"]) == d["rho_grid"]
        assert r_dp["lam"] == d["lam"]
        assert r_dp["tol"] == d["tol"]
        assert [X.shape[0] for X, _ in r_dp["cohort"]] == list(d["n_per"])

    def test_surrogate_sweep_matches_R(self, r_dp):
        # Iteration counts are the first integer at which a continuous
        # residual crosses 1e-3, so a one-off difference would not be a
        # defect. Measured, they agree exactly: 60/60/33/28/60.
        choice = dp.choose_rho_and_T(r_dp["surrogate"])
        ref = r_dp["ref"]["rho_sweep"]
        assert [r["converged"] for r in choice.sweep] == ref["converged"]
        for got, want in zip(choice.sweep, ref["iters"], strict=True):
            assert abs(got["iters"] - want) <= 2
        assert choice.rho == r_dp["ref"]["rho_chosen"]
        assert abs(choice.T - r_dp["ref"]["T_fixed"]) <= 2

    def test_centralized_fit_matches_R(self, r_dp):
        beta = centralized(r_dp["cohort"], r_dp["lam"])
        np.testing.assert_allclose(
            beta, r_dp["ref"]["beta_centralized"], atol=STATISTICAL_TOL
        )

    def test_noiseless_protocol_reaches_the_centralized_fit(self, r_dp):
        ref = r_dp["ref"]
        beta_c = np.asarray(ref["beta_centralized"], dtype=float)
        fit = dp.consensus_at_sigma(
            0.0, r_dp["cohort"], ref["rho_chosen"], ref["T_fixed"], beta_c
        )
        assert fit.max_dev < 10 * r_dp["tol"]
        # R's own sigma = 0 deviation, to stopping-tolerance scale.
        assert abs(fit.max_dev - ref["clean_dev"]) < STATISTICAL_TOL


class TestOnRsEarlierCohort:
    """The ADMM machinery on a second R draw (``admm_cohort``)."""

    @pytest.fixture(scope="class")
    @staticmethod
    def cohort():
        from homomorphepy import load_json

        f = load_json("admm_cohort")
        return _sites(f["sites"], f["p"]), f["lambda"]

    def test_plaintext_admm_matches_centralized(self, cohort):
        data, lam = cohort
        rho, p = 50.0, data[0][0].shape[1]
        sites = [
            dp.ConsensusSite(f"Site {i + 1}", X, y, rho, lam, len(data))
            for i, (X, y) in enumerate(data)
        ]
        z, _, _, ok = admm_loop(sites, p, rho, 60, 1e-3, plain_consensus)
        assert ok
        assert np.max(np.abs(z - centralized(data, lam))) < STATISTICAL_TOL

    def test_encrypted_matches_plaintext(self, cohort):
        data, lam = cohort
        rho, p = 50.0, data[0][0].shape[1]
        build = lambda: [  # noqa: E731
            dp.ConsensusSite(f"Site {i + 1}", X, y, rho, lam, len(data))
            for i, (X, y) in enumerate(data)
        ]
        z_plain, T, _, _ = admm_loop(build(), p, rho, 60, 1e-3, plain_consensus)
        fit = dp.consensus_at_sigma(0.0, data, rho, T, z_plain)
        # max_dev is measured against what was passed in: here the
        # plaintext run on the same data, for the same T.
        assert fit.max_dev < CKKS_TOL


class TestOnAFreshDraw:
    """The example's claims, on data numpy drew."""

    @pytest.fixture(scope="class")
    @staticmethod
    def fresh():
        d = dp.CONSENSUS_DGP
        cohort = dp.simulate_cohort(d["beta_true"], seed=7)
        surrogate = dp.simulate_cohort(d["beta_nominal"], seed=8)
        return cohort, surrogate

    def test_design(self, fresh):
        cohort, _ = fresh
        X, y = cohort[0]
        assert X.shape == (500, 4)
        assert np.all(X[:, 0] == 1.0)
        assert set(np.unique(X[:, 3])) <= {0.0, 1.0}
        assert set(np.unique(y)) <= {0.0, 1.0}

    def test_surrogate_step_picks_a_converging_rho(self, fresh):
        _, surrogate = fresh
        choice = dp.choose_rho_and_T(surrogate)
        chosen = [r for r in choice.sweep if r["rho"] == choice.rho][0]
        assert chosen["converged"]
        assert choice.T == chosen["iters"] < dp.CONSENSUS_DGP["max_iter"]

    def test_noiseless_protocol_reaches_the_centralized_fit(self, fresh):
        cohort, surrogate = fresh
        res = dp.consensus_sweep(sigmas=(0.0,), cohort=cohort, surrogate=surrogate)
        assert res.clean_dev < 10 * res.tol


@pytest.mark.slow
class TestNoise:
    """The noisy sweep. Behavior only; values are not reproducible."""

    @pytest.fixture(scope="class")
    @staticmethod
    def result():
        return dp.consensus_sweep()

    def test_deviation_grows_with_sigma(self, result):
        by_sigma = {f.sigma: f.max_dev for f in result.fits}
        assert by_sigma[1e-4] < by_sigma[1e-1] < by_sigma[1.0]

    def test_runs_exactly_T_iterations(self, result):
        for f in result.fits:
            assert len(f.trajectory) == result.choice.T

    def test_budget_counts_T_releases_without_a_factor_of_N(self, result):
        # One Gaussian release of the average per iteration, whatever
        # the number of sites.
        b = dp.budget(result.choice.T, sigma=0.1)
        assert b["rho_total"] == pytest.approx(result.choice.T * (1 / 0.1) ** 2 / 2)
        assert math.isfinite(b["epsilon"])


class TestSolverDiscipline:
    """The solver is named, never inherited from an automatic choice."""

    def test_solver_is_clarabel(self):
        assert SOLVER == "CLARABEL"

    def test_solver_is_in_the_portable_set(self):
        assert SOLVER in SUPPORTED_SOLVERS

    def test_pinned_solver_is_installed(self):
        import cvxpy as cp

        assert "CLARABEL" in set(cp.installed_solvers())
