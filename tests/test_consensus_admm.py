"""Consensus ADMM: the claims, tested across draws rather than one.

The example simulates its own cohort with numpy by default. Matching
R's exact random draw is loaded here (see ``_r_cohort``) but reserved for
cross-language value comparison; the scientific claims should hold for
any draw from the data-generating process, and that is what is checked
here.

Measured across R's draw plus four numpy draws:

    draw            rho   plain/enc iters   |enc-plain|   |enc-central|
    R's exact       50    19/19             8.8e-08       5.4e-06
    numpy 98765     50    21/21             5.8e-07       9.5e-06
    numpy 1         50    23/23             2.1e-07       1.4e-05
    numpy 2         50    23/23             1.9e-06       1.2e-05
    numpy 7         50    19/19             1.9e-07       9.2e-06

Note the iteration count moving 19 -> 23 *within* Python purely by
redrawing the data. It is a property of the sample, not of the
language, so an exact cross-language iteration match was never a
meaningful target.
"""

from __future__ import annotations

import numpy as np
import pytest

from homomorphepy import have_backend, set_thread_env

set_thread_env(2)

pytest.importorskip("cvxpy", reason="install the 'stats' extra")


def _r_cohort():
    """The exact cohort R drew, for the cross-language check below.

    Lives in the test rather than the package: homomorphepy does not
    need R to be a concept in its public API, but our own parity
    checking does need R's draw.
    """
    from homomorphepy import load_json

    f = load_json("admm_cohort")
    p = f["p"]
    return [
        (
            np.asarray(s["X"], dtype=float).reshape(s["n"], p),
            np.asarray(s["y"], dtype=float),
        )
        for s in f["sites"]
    ]


pytestmark = [
    pytest.mark.openfhe,
    pytest.mark.skipif(not have_backend(), reason="openfhe not installed"),
]

# Encrypted vs plaintext on identical data: the cryptographic claim.
CKKS_TOL = 1e-5
# ADMM iterate vs the centralized optimum: convergence-tolerance scale
# (the loop stops at primal/dual residual 1e-3), not crypto scale.
STATISTICAL_TOL = 1e-4

SEEDS = (98765, 1, 7)


@pytest.fixture(scope="module")
def runs():
    from homomorphepy.examples import consensus_admm as admm

    return {seed: admm.run(seed=seed) for seed in SEEDS}


class TestCryptographicClaim:
    """Encrypting the consensus step changes nothing but precision."""

    def test_encrypted_matches_plaintext_on_every_draw(self, runs):
        for seed, r in runs.items():
            assert r.max_abs_encrypted_vs_plaintext < CKKS_TOL, f"seed {seed}"

    def test_both_runs_converge(self, runs):
        from homomorphepy.examples.consensus_admm import DGP

        for seed, r in runs.items():
            assert r.n_iter_encrypted < DGP["max_iter"], f"seed {seed}"
            assert r.n_iter_plaintext < DGP["max_iter"], f"seed {seed}"

    def test_encryption_does_not_degrade_convergence_rate(self, runs):
        # Deliberately NOT an equality. The loop stops when the primal
        # and dual residuals first fall below an absolute 1e-3, so the
        # iteration count is the first integer at which a continuous
        # quantity crosses a fixed threshold -- a step function whose
        # value flips under any perturbation when a residual passes
        # close to a boundary. CKKS noise at ~1e-6 is ample, and so
        # would be a different BLAS summation order.
        #
        # Measured on identical data with fresh threshold keys per run:
        # seed 1 gives (plain, enc) of (23,24), (23,23), (23,24),
        # (23,24) -- the encrypted count moves while the answer does
        # not. Seed 2 gives (23,23) every time; it simply is not near a
        # boundary.
        #
        # What would matter is encryption systematically slowing
        # convergence, e.g. noise preventing the residuals from
        # settling. A generous bound catches that while ignoring which
        # side of a threshold the last iterate landed on.
        for seed, r in runs.items():
            assert abs(r.n_iter_encrypted - r.n_iter_plaintext) <= 2, (
                f"seed {seed}: {r.n_iter_plaintext} -> {r.n_iter_encrypted}"
            )


class TestStatisticalClaim:
    """The federated fit recovers the centralized one, on any draw."""

    def test_matches_centralized_on_every_draw(self, runs):
        for seed, r in runs.items():
            assert r.max_abs_vs_centralized < STATISTICAL_TOL, f"seed {seed}"

    def test_recovers_the_true_coefficients_approximately(self, runs):
        from homomorphepy.examples.consensus_admm import DGP

        beta_true = np.asarray(DGP["beta_true"], dtype=float)
        for seed, r in runs.items():
            # Sampling error at n ~ 1000, not a numerical claim: loose
            # on purpose.
            assert np.max(np.abs(r.beta_encrypted - beta_true)) < 0.25, f"seed {seed}"


class TestTuningIsRobust:
    def test_same_rho_chosen_on_every_draw(self, runs):
        # rho = 50 is selected from {10, 50, 200} regardless of the
        # draw, and matches R's choice. The tuning is a property of the
        # problem, not of one dataset.
        assert {r.rho for r in runs.values()} == {50.0}

    def test_largest_rho_fails_to_converge(self, runs):
        # rho = 200 exceeds max_iter, so the sweep records None. R uses
        # which.min(), which drops NA; np.argmin would return the NaN's
        # index and select this, the WORST rho. Trap flagged in review
        # and avoided by filtering before the min.
        for r in runs.values():
            assert r.rho_sweep[200.0] is None
            assert r.rho != 200.0


class TestSolverDiscipline:
    """The solver is named, never inherited from an automatic choice."""

    def test_solver_is_clarabel_as_in_R(self, runs):
        from homomorphepy.examples.consensus_admm import SOLVER

        assert SOLVER == "CLARABEL"
        for r in runs.values():
            assert r.solver == "CLARABEL"

    def test_solver_is_in_the_interoperable_set(self):
        from homomorphepy.examples.consensus_admm import SOLVER, SUPPORTED_SOLVERS

        # Clarabel / SCS / HiGHS / OSQP ship in both cvxpy and CVXR.
        # A solver outside this set has no R counterpart, so a ported
        # example could not be run on the other side at all.
        assert SOLVER in SUPPORTED_SOLVERS

    def test_every_supported_solver_is_installed_or_absent_knowingly(self):
        import cvxpy as cp

        from homomorphepy.examples.consensus_admm import SUPPORTED_SOLVERS

        installed = set(cp.installed_solvers())
        assert "CLARABEL" in installed, "the pinned solver must be present"
        # Not a requirement, just visibility: record which of the
        # interoperable set this environment actually has.
        assert set(SUPPORTED_SOLVERS) & installed


class TestAgreesWithR:
    """Cross-language, on R's exact draw. Value comparison only."""

    @pytest.fixture(scope="class")
    @staticmethod
    def r_run():
        from homomorphepy.examples import consensus_admm as admm

        return admm.run(cohort=_r_cohort())

    def test_encrypted_matches_plaintext_on_Rs_data_too(self, r_run):
        assert r_run.max_abs_encrypted_vs_plaintext < CKKS_TOL

    def test_matches_centralized_on_Rs_data(self, r_run):
        assert r_run.max_abs_vs_centralized < STATISTICAL_TOL

    def test_same_rho_as_R(self, r_run):
        assert r_run.rho == 50.0
