"""Consensus ADMM machinery shared by the DP and Cox-lasso examples.

Private: :mod:`.dp` uses all of it for L2-regularized logistic
regression under output DP, and :mod:`.cox_lasso` uses :data:`SOLVER`.

With ``N`` sites and a shared coefficient ``x``, the global problem

    min_x  sum_i l_i(x; X_i, y_i) + (lam/2) ||x||^2

splits into local copies ``x_i`` tied to a consensus ``z``. Each
iteration solves a local problem at every site, averages
``x_i + u_i`` across sites to get ``z``, and updates each site's dual
``u_i``. Only the average leaves a site, so only the average needs
encryption.

Solver discipline
-----------------

**The solver is named explicitly, never left to the modeling layer's
automatic choice.** Every ``problem.solve()`` here passes
``solver=cp.CLARABEL``. Two reasons:

1. cvxpy's preference order among installed solvers is its own, so
   the automatic choice can change the algorithm without any change
   to the code.
2. The automatic choice depends on *what happens to be installed*, so
   the same script can change behavior on a different machine with no
   diff to point at.

:data:`SOLVER` is the single place it is set, and
:data:`SUPPORTED_SOLVERS` records the portable set -- Clarabel, SCS,
HiGHS and OSQP.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import cvxpy as cp
import numpy as np

from homomorphepy.actors import Site

__all__ = [
    "CKKS_PARAMS",
    "SOLVER",
    "SUPPORTED_SOLVERS",
    "ConsensusSite",
    "admm_loop",
    "centralized",
    "plain_consensus",
]

# Conic/QP solvers that are widely packaged; anything else is a
# machine-specific choice.
SUPPORTED_SOLVERS = ("CLARABEL", "SCS", "HIGHS", "OSQP")

# Named explicitly rather than inherited from cvxpy's preference
# order -- see the module docstring.
SOLVER = cp.CLARABEL

CKKS_PARAMS = dict(
    multiplicative_depth=1, scaling_mod_size=59, first_mod_size=60, batch_size=8
)


class ConsensusSite(Site):
    """One ADMM peer: a local convex problem plus its ADMM state.

    A :class:`~homomorphepy.actors.Site`, so it takes part in threshold
    key generation and keeps its own share, and encrypts its own
    ``x_i + u_i`` with :meth:`~homomorphepy.actors.Site.encrypt`. It
    has no ``local_fn``: ADMM does not use the master/worker
    ``contribute`` round, and the site's state is its iterates.

    ``z`` and ``u`` are cvxpy Parameters so the problem canonicalizes
    once and every iteration reuses it; ``X``, ``y`` and ``rho`` are
    baked in as constants so the augmented Lagrangian stays affine in
    the parameters and DPP's fast path is not broken.
    """

    _needs_local_fn = False

    def __init__(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        rho: float,
        lam: float,
        n_sites: int,
    ):
        super().__init__(name, (X, y), None)
        self.n = X.shape[0]
        p = X.shape[1]

        self.x_var = cp.Variable(p)
        self.z_par = cp.Parameter(p, value=np.zeros(p))
        self.u_par = cp.Parameter(p, value=np.zeros(p))

        signs = 2.0 * y - 1.0
        margins = -cp.multiply(signs, X @ self.x_var)
        local_loss = cp.sum(cp.logistic(margins)) + (
            lam / (2 * n_sites)
        ) * cp.sum_squares(self.x_var)
        augmented = (rho / 2) * cp.sum_squares(self.x_var - self.z_par + self.u_par)
        self.problem = cp.Problem(cp.Minimize(local_loss + augmented))

        self.x_curr = np.zeros(p)
        self.u_curr = np.zeros(p)

    def contribute(self, theta):  # pragma: no cover - not part of ADMM
        raise NotImplementedError(
            f"{self.name!r} is an ADMM peer; it takes part through "
            "local_update() and dual_update(), not contribute()"
        )

    def local_update(self, z_curr: np.ndarray) -> np.ndarray:
        """Solve the local subproblem at the current consensus."""
        self.z_par.value = np.asarray(z_curr, dtype=float)
        self.u_par.value = self.u_curr
        with warnings.catch_warnings():
            # CLARABEL reports optimal_inaccurate on some ADMM
            # subproblems and cvxpy warns per solve, which would fire
            # on most iterations. The status is checked explicitly
            # below.
            warnings.simplefilter("ignore", UserWarning)
            self.problem.solve(solver=SOLVER)
        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(
                f"local solve at {self.name} ended {self.problem.status!r}"
            )
        self.x_curr = np.asarray(self.x_var.value, dtype=float).ravel()
        return self.x_curr

    def dual_update(self, z_new: np.ndarray) -> None:
        self.u_curr = self.u_curr + (self.x_curr - z_new)

    def __repr__(self) -> str:
        return f"<ConsensusSite {self.name} n={self.n}>"


def plain_consensus(sites: Sequence[ConsensusSite]) -> np.ndarray:
    """The consensus average computed in the clear."""
    return sum(s.x_curr + s.u_curr for s in sites) / len(sites)


def centralized(cohort, lam: float) -> np.ndarray:
    """The pooled fit of the global problem, for comparison only."""
    X = np.vstack([s[0] for s in cohort])
    y = np.concatenate([s[1] for s in cohort])
    beta = cp.Variable(X.shape[1])
    margins = -cp.multiply(2.0 * y - 1.0, X @ beta)
    prob = cp.Problem(
        cp.Minimize(cp.sum(cp.logistic(margins)) + (lam / 2) * cp.sum_squares(beta))
    )
    prob.solve(solver=SOLVER)
    return np.asarray(beta.value, dtype=float).ravel()


def admm_loop(
    sites: Sequence[ConsensusSite],
    p: int,
    rho: float,
    max_iter: int,
    tol: float,
    consensus_fn: Callable[[Sequence[ConsensusSite]], np.ndarray],
):
    """The ADMM iteration. ``consensus_fn`` averages the ``x + u`` vectors.

    Identical whether that average is computed in the clear or through
    the encrypted channel. Stops when the primal residual
    ``sqrt(mean_i ||x_i - z||^2)`` and the dual residual
    ``rho * ||z - z_prev||`` are both below ``tol``; ``tol = 0`` runs
    exactly ``max_iter`` iterations.

    Returns ``(z, iterations, trajectory, converged)``; ``converged``
    is whether the residual test fired, which ``iterations`` alone
    cannot say when it fires on the last iteration.
    """
    z = np.zeros(p)
    trajectory = []
    for k in range(1, max_iter + 1):
        for s in sites:
            s.local_update(z)
        z_new = consensus_fn(sites)
        for s in sites:
            s.dual_update(z_new)

        primal = float(
            np.sqrt(np.mean([np.sum((s.x_curr - z_new) ** 2) for s in sites]))
        )
        dual = float(rho * np.linalg.norm(z_new - z))
        z = z_new
        trajectory.append(z.copy())
        if primal < tol and dual < tol:
            return z, k, trajectory, True
    return z, max_iter, trajectory, False
