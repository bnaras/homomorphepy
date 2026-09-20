"""Consensus ADMM over a threshold-encrypted channel.

Three sites fit a
shared logistic model by ADMM: each solves a local convex subproblem,
and the only step that leaves a site is the consensus average

    z^{k+1} = (1/N) sum_i (x_i^{k+1} + u_i^k)

which travels encrypted. Each site packs its length-p vector into one
encrypted value under the joint public key; the aggregator sums those
and multiplies by the unencrypted constant 1/N (one multiplication by
a cleartext value, one level of the precision budget) and the sites jointly
threshold-decrypt the result.

The aggregator therefore learns the consensus trajectory {z^k} and
nothing else: the per-site (x_i + u_i) vectors never exist in
in the clear outside their own site, and no party can decrypt alone.

Solver discipline
-----------------

**The solver is named explicitly, never left to the modeling layer's
automatic choice.** R passes ``solver = "CLARABEL"`` to every
``psolve()`` call; this module passes ``solver=cp.CLARABEL`` to every
``problem.solve()``. Two reasons, and the second is the serious one:

1. cvxpy and CVXR rank the installed solvers differently, so the
   automatic choice can silently differ across languages -- and then a
   trajectory difference looks like a bug in the protocol rather than
   a difference of algorithm.
2. The automatic choice depends on *what happens to be installed*, so
   the same script can change behavior on a different machine with no
   diff to point at.

:data:`SOLVER` is the single place it is set, and
:data:`SUPPORTED_SOLVERS` records the interoperable set -- Clarabel,
SCS, HiGHS and OSQP -- that both ecosystems ship. Anything outside
that list has no counterpart on the R side.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from homomorphepy.actors import ThresholdMaster, make_threshold_master, make_worker
from homomorphepy.context import Context, fhe_context

__all__ = [
    "DGP",
    "SOLVER",
    "SUPPORTED_SOLVERS",
    "ADMMResult",
    "ConsensusSite",
    "run",
    "simulate",
]

# The conic/QP solvers available in BOTH cvxpy and CVXR. Restricting to
# these keeps a ported example runnable on either side; anything else
# has no R counterpart and would break the comparison.
SUPPORTED_SOLVERS = ("CLARABEL", "SCS", "HIGHS", "OSQP")

# What R uses. Named explicitly rather than inherited from cvxpy's
# preference order -- see the module docstring.
SOLVER = cp.CLARABEL

CKKS_PARAMS = dict(
    multiplicative_depth=1, scaling_mod_size=59, first_mod_size=60, batch_size=8
)


class ConsensusSite:
    """One ADMM peer: a local convex problem plus its ADMM state.

    Separate from :class:`~homomorphepy.actors.Site`, which is shaped
    for master/worker fan-in. ADMM is peer-to-peer and carries its own
    per-site iterates, so it defines its own site class rather than
    reusing the exported one.

    ``z`` and ``u`` are cvxpy Parameters so the problem canonicalizes
    once and every iteration reuses it; ``X``, ``y`` and ``rho`` are
    baked in as constants so the augmented Lagrangian stays affine in
    the parameters and DPP's fast path is not broken.
    """

    def __init__(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        rho: float,
        lam: float,
        n_sites: int,
    ):
        self.name = name
        self.X = X
        self.y = y
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

    def local_update(self, z_curr: np.ndarray) -> np.ndarray:
        """Solve the local subproblem at the current consensus."""
        self.z_par.value = np.asarray(z_curr, dtype=float)
        self.u_par.value = self.u_curr
        with warnings.catch_warnings():
            # CLARABEL reports optimal_inaccurate on some ADMM
            # subproblems and cvxpy warns per solve, which would fire
            # ~60 times per run. The status is checked explicitly
            # below, and R accepts the same one.
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


@dataclass
class ADMMResult:
    beta_encrypted: np.ndarray
    beta_plaintext: np.ndarray
    beta_centralized: np.ndarray
    n_iter_encrypted: int
    n_iter_plaintext: int
    rho: float
    rho_sweep: dict[float, int | None]
    trajectory: list[np.ndarray]
    solver: str
    max_abs_vs_centralized: float
    max_abs_encrypted_vs_plaintext: float
    context: Context = field(repr=False)
    master: ThresholdMaster = field(repr=False)


# The data-generating process, not one draw of it. beta_true, the
# site sizes and the penalty are fixed; the draw does
# not have to.
DGP = dict(
    p=4,
    beta_true=(0.5, -1.0, 0.3, 0.8),
    n_per=(400, 250, 350),
    lam=1.0,
    rho_grid=(10.0, 50.0, 200.0),
    max_iter=40,
    tol=1e-3,
)


def simulate(seed: int = 98765):
    """Draw a cohort of three sites from the DGP.

    A fresh draw per seed. The claim the example makes is that the
    encrypted fit reproduces the centralized fit *on the data at hand*,
    which should hold for any draw; ``tests/test_consensus_admm.py``
    checks exactly that across several seeds.
    """
    rng = np.random.default_rng(seed)
    beta = np.asarray(DGP["beta_true"], dtype=float)
    sites = []
    for n in DGP["n_per"]:
        X = rng.normal(size=(n, DGP["p"]))
        prob = 1.0 / (1.0 + np.exp(-(X @ beta)))
        sites.append((X, rng.binomial(1, prob).astype(float)))
    return sites


def _centralized(sites, lam: float) -> np.ndarray:
    X = np.vstack([s[0] for s in sites])
    y = np.concatenate([s[1] for s in sites])
    beta = cp.Variable(X.shape[1])
    margins = -cp.multiply(2.0 * y - 1.0, X @ beta)
    prob = cp.Problem(
        cp.Minimize(cp.sum(cp.logistic(margins)) + (lam / 2) * cp.sum_squares(beta))
    )
    prob.solve(solver=SOLVER)
    return np.asarray(beta.value, dtype=float).ravel()


def _admm_loop(sites, p, rho, max_iter, tol, consensus_fn):
    """The ADMM iteration. ``consensus_fn`` averages the (x+u) vectors.

    Identical whether that average is computed in the clear or through
    the encrypted channel -- which is the whole point of the example.
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
        dual = float(np.sqrt(len(sites)) * rho * np.linalg.norm(z_new - z))
        z = z_new
        trajectory.append(z.copy())
        if primal < tol and dual < tol:
            return z, k, trajectory
    return z, max_iter, trajectory


def run(seed: int = 98765, cohort=None) -> ADMMResult:
    """Fit the consensus logistic model, unencrypted and encrypted.

    Simulates a cohort from the DGP unless one is passed in as
    ``cohort``, a sequence of ``(X, y)`` pairs -- one per site.
    """
    if cohort is None:
        cohort = simulate(seed)
    p, N, lam = DGP["p"], len(cohort), DGP["lam"]
    tol, max_iter = DGP["tol"], DGP["max_iter"]

    def build(rho):
        return [
            ConsensusSite(f"Site {i + 1}", X, y, rho, lam, N)
            for i, (X, y) in enumerate(cohort)
        ]

    def plain_consensus(sites):
        return sum(s.x_curr + s.u_curr for s in sites) / len(sites)

    # -- rho sweep, in the clear ---------------------------------------
    sweep: dict[float, int | None] = {}
    for rho in DGP["rho_grid"]:
        sites = build(rho)
        _, k, _ = _admm_loop(sites, p, rho, max_iter, tol, plain_consensus)
        sweep[float(rho)] = None if k >= max_iter else k

    # R picks with which.min(), which IGNORES NA. numpy's argmin would
    # return the NaN's index and select the worst rho -- a porting trap
    # flagged in review, avoided here by filtering first.
    converged = {r: k for r, k in sweep.items() if k is not None}
    if not converged:
        raise RuntimeError("no rho in the grid converged")
    rho = min(converged, key=lambda r: converged[r])

    # -- unencrypted reference at the chosen rho ----------------------
    sites = build(rho)
    beta_plain, n_plain, _ = _admm_loop(sites, p, rho, max_iter, tol, plain_consensus)

    # -- encrypted run: same loop, encrypted consensus ----------------
    ctx = fhe_context("CKKS", **CKKS_PARAMS)
    enc_sites = build(rho)
    # The key-holding parties; the ADMM peers are separate objects, so
    # pair them by position.
    key_holders = [
        make_worker(f"Site {i + 1}", None, lambda d, t: 0.0) for i in range(N)
    ]
    master = make_threshold_master("Aggregator", ctx, key_holders)

    # What each party kept from that one exchange: its own secret share
    # and a copy of the public parameters — the crypto context and the
    # joint public key, and no share of anyone else's. Asking a site
    # what it holds involves no aggregator.
    pub = [h.params for h in key_holders]

    def encrypted_consensus(sites):
        # Each party encrypts its own x_k + u_k with the parameters it
        # kept from wiring. Encrypting at the aggregator instead would
        # mean handing it the per-site vectors in the clear first,
        # which is the disclosure this round exists to avoid.
        cts = [
            par.encrypt(s.x_curr + s.u_curr) for par, s in zip(pub, sites, strict=True)
        ]
        ct_avg = sum(cts) * (1.0 / len(sites))
        return np.asarray(master.decrypt(ct_avg, length=p), dtype=float)

    beta_enc, n_enc, trajectory = _admm_loop(
        enc_sites, p, rho, max_iter, tol, encrypted_consensus
    )

    beta_central = _centralized(cohort, lam)

    return ADMMResult(
        beta_encrypted=beta_enc,
        beta_plaintext=beta_plain,
        beta_centralized=beta_central,
        n_iter_encrypted=n_enc,
        n_iter_plaintext=n_plain,
        rho=float(rho),
        rho_sweep=sweep,
        trajectory=trajectory,
        solver=str(SOLVER),
        max_abs_vs_centralized=float(np.max(np.abs(beta_enc - beta_central))),
        max_abs_encrypted_vs_plaintext=float(np.max(np.abs(beta_enc - beta_plain))),
        context=ctx,
        master=master,
    )


if __name__ == "__main__":  # pragma: no cover
    r = run()
    print(f"solver            : {r.solver} (explicit, not auto-selected)")
    print("cohort            : simulated fresh from the DGP")
    print(f"rho sweep         : {r.rho_sweep}  -> chose {r.rho:g}")
    print(
        f"iterations        : plaintext {r.n_iter_plaintext}, "
        f"encrypted {r.n_iter_encrypted}"
    )
    print(f"beta encrypted    : {np.round(r.beta_encrypted, 6)}")
    print(f"beta centralized  : {np.round(r.beta_centralized, 6)}")
    print(f"max |enc - plain| : {r.max_abs_encrypted_vs_plaintext:.3e}")
    print(f"max |enc - central|: {r.max_abs_vs_centralized:.3e}")
