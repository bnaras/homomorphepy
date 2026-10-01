"""Cox-lasso on DLBCL gene expression, federated under threshold FHE.

The largest example in the set, and the only one where every stage of
the pipeline needs the encrypted channel:

1. **Standardization.** Column means and second moments are pooled
   across sites under encryption, so no site's marginal distribution
   is revealed.
2. **Screening.** 6416 probes are ranked by a univariate Cox score at
   beta = 0, again pooled under encryption, and the top K = 100 kept.
3. **Consensus ADMM.** The lasso-penalized stratified Cox fit is
   reached by ADMM, with the consensus average traversing the
   encrypted channel at every iteration. The same loop runs twice, first with an
   unencrypted average and then with the encrypted one, so the
   difference between the two is the encryption's contribution.

**The tie convention is Breslow here, not Efron.** The partial
likelihood is built symbolically as ``log_sum_exp(eta[R_i]) - eta[i]``
over event times, with ``R_i`` every row whose time is at least
``t_i``, so cvxpy can canonicalize it; that expression *is* the
Breslow form. Efron has no comparable convex-atom formulation.
So this example and :mod:`.cox` deliberately use different
conventions -- Efron there via PHReg, Breslow here by construction.

**Measured data, so it ships with the package.** Unlike the simulated
examples there is no draw to repeat: the DLBCL cohort is what it is,
and every run reads the same bytes.

Expensive: each ADMM iteration makes three per-site conic solves, and
each run takes over a hundred iterations. ``run(recompute_admm=False)`` performs the standardization
and screening -- the parts that exercise the encrypted channel most
interestingly -- and stops short of the loop.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from homomorphepy.actors import ThresholdMaster, make_threshold_master, make_worker
from homomorphepy.context import Context, fhe_context
from homomorphepy.examples._consensus import SOLVER
from homomorphepy.fixtures import load_dlbcl, load_dlbcl_gex, site_order

__all__ = [
    "CoxLassoResult",
    "K",
    "LAMBDA",
    "RHO",
    "build_local",
    "plain_consensus",
    "run",
    "run_admm",
    "soft_threshold",
]

K = 100
LAMBDA = 5.0
RHO = 50.0
MAX_ITER = 200
TOL = 5e-3

# Depth 1 suffices: the encrypted stages are sums and one scalar
# multiply. batch_size must hold a full 6416-length vector for the
# screening stage, so it is far larger than the other examples'.
CKKS_PARAMS = dict(
    multiplicative_depth=1, scaling_mod_size=59, first_mod_size=60, batch_size=8192
)


# A coefficient counts as selected above this magnitude.
NONZERO = 1e-7


@dataclass
class CoxLassoResult:
    top_idx: np.ndarray
    sigma: np.ndarray
    pool_agree_mu: float
    pool_agree_sigma: float
    # Whether the encrypted screen kept the same probes as the same
    # screen computed in the clear.
    screen_match: bool
    beta_centralized: np.ndarray | None
    # The ADMM run with an ordinary unencrypted average, and the same
    # run with the encrypted one.
    beta_plain: np.ndarray | None
    n_iter_plain: int | None
    beta_admm: np.ndarray | None
    n_iter: int | None
    # The encrypted consensus iterate after every ADMM sweep, so the
    # run can be plotted as trajectories rather than only as its
    # endpoint. One row per iteration, K columns; empty when the ADMM
    # was skipped.
    trajectory: list[np.ndarray]
    context: Context = field(repr=False)
    master: ThresholdMaster = field(repr=False)

    @property
    def n_nonzero(self) -> int | None:
        """How many of the K screened probes the encrypted fit retained."""
        if self.beta_admm is None:
            return None
        return int(np.sum(np.abs(self.beta_admm) > NONZERO))

    @property
    def n_nonzero_centralized(self) -> int | None:
        if self.beta_centralized is None:
            return None
        return int(np.sum(np.abs(self.beta_centralized) > NONZERO))

    @property
    def n_active_intersection(self) -> int | None:
        """Probes selected by both the centralized and the encrypted fit."""
        if self.beta_admm is None or self.beta_centralized is None:
            return None
        return int(
            np.sum(
                (np.abs(self.beta_admm) > NONZERO)
                & (np.abs(self.beta_centralized) > NONZERO)
            )
        )

    @property
    def admm_vs_plain(self) -> float | None:
        """Max abs difference: encrypted ADMM against unencrypted ADMM.

        Same loop, same data, same stopping rule; only the average is
        encrypted. This is the CKKS approximation error in the fit.
        """
        if self.beta_admm is None or self.beta_plain is None:
            return None
        return float(np.max(np.abs(self.beta_admm - self.beta_plain)))

    @property
    def plain_vs_centralized(self) -> float | None:
        """Max abs difference: unencrypted ADMM against the centralized fit.

        Set by stopping ADMM at a finite tolerance, not by encryption.
        """
        if self.beta_plain is None or self.beta_centralized is None:
            return None
        return float(np.max(np.abs(self.beta_plain - self.beta_centralized)))

    @property
    def admm_vs_centralized(self) -> float | None:
        """Max abs difference: the encrypted ADMM fit against the centralized one."""
        if self.beta_admm is None or self.beta_centralized is None:
            return None
        return float(np.max(np.abs(self.beta_admm - self.beta_centralized)))


def _sites_raw():
    """Per-site (X, time, status) on the raw expression scale."""
    df = load_dlbcl()
    gex, _, _ = load_dlbcl_gex()
    out = []
    for name in site_order():
        m = (df["Subgroup"] == name).to_numpy()
        out.append(
            {
                "name": name,
                "X": gex[m],
                "time": df["time"].to_numpy(dtype=float)[m],
                "status": df["status"].to_numpy(dtype=int)[m],
            }
        )
    return out


def _score_info_at_zero(X, time, status):
    """Univariate Cox score and information at beta = 0, per column.

    The risk set at event ``i`` is every row with ``time >= t_i``,
    tied rows included.
    """
    order = np.argsort(time, kind="stable")
    Xo, so, to = X[order], status[order], time[order]
    first = np.searchsorted(to, to, side="left")
    n, p = Xo.shape
    U = np.zeros(p)
    info = np.zeros(p)
    for i in range(n):
        if so[i] == 1:
            risk = Xo[first[i]:]
            mu = risk.mean(axis=0)
            U += Xo[i] - mu
            info += ((risk - mu) ** 2).sum(axis=0) / risk.shape[0]
    return U, info


def _breslow_nll(beta, X, time, status):
    """Cox partial NLL as a cvxpy expression (Breslow ties).

    ``sum_j in events [ log_sum_exp(eta_{R_j}) - eta_j ]``, where the
    risk set ``R_j`` is every row with ``time >= t_j``, tied rows
    included. This is exactly the Breslow partial likelihood; Efron has
    no equivalent convex-atom form, which is why this example and
    :mod:`.cox` use different tie conventions.
    """
    order = np.argsort(time, kind="stable")
    Xo, so, to = X[order], status[order], time[order]
    first = np.searchsorted(to, to, side="left")
    eta = Xo @ beta
    terms = [
        cp.log_sum_exp(eta[first[i]:]) - eta[i] for i in range(len(so)) if so[i] == 1
    ]
    return cp.sum(terms) if len(terms) > 1 else terms[0]


def _solve(problem):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        problem.solve(solver=SOLVER, verbose=False)


def soft_threshold(v: np.ndarray, tau: float) -> np.ndarray:
    """The proximal map of ``tau * ||.||_1``."""
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)


def build_local(X, time, status, rho: float) -> dict:
    """One site's ADMM subproblem, built once.

    ``z`` and ``u`` are cvxpy Parameters, so the problem canonicalizes
    once and every iteration only changes their values.
    """
    p = X.shape[1]
    x = cp.Variable(p)
    zp = cp.Parameter(p, value=np.zeros(p))
    up = cp.Parameter(p, value=np.zeros(p))
    obj = _breslow_nll(x, X, time, status) + (rho / 2) * cp.sum_squares(x - zp + up)
    return {"prob": cp.Problem(cp.Minimize(obj)), "x": x, "zp": zp, "up": up}


def run_admm(locals_, consensus):
    """The consensus-ADMM driver.

    ``consensus(site_x, site_u)`` returns the average of the per-site
    ``x_k + u_k`` vectors. Called once with an unencrypted average and
    once, unchanged, with an encrypted one. Returns ``(z, trajectory)``.
    """
    n_sites = len(locals_)
    site_x = [np.zeros(K) for _ in locals_]
    site_u = [np.zeros(K) for _ in locals_]
    z_curr = np.zeros(K)
    trajectory: list[np.ndarray] = []
    for _ in range(MAX_ITER):
        for i, lp in enumerate(locals_):
            lp["zp"].value = z_curr
            lp["up"].value = site_u[i]
            _solve(lp["prob"])
            site_x[i] = np.asarray(lp["x"].value, dtype=float).ravel()
        w_avg = consensus(site_x, site_u)
        z_new = soft_threshold(w_avg, LAMBDA / (n_sites * RHO))
        site_u = [u + (x - z_new) for u, x in zip(site_u, site_x, strict=True)]
        primal = float(np.sqrt(np.mean([np.sum((x - z_new) ** 2) for x in site_x])))
        dual = float(RHO * np.linalg.norm(z_new - z_curr))
        z_curr = z_new
        trajectory.append(z_curr.copy())
        if primal < TOL and dual < TOL:
            break
    return z_curr, trajectory


def plain_consensus(site_x, site_u) -> np.ndarray:
    """The consensus average, computed in the clear."""
    return sum(x + u for x, u in zip(site_x, site_u, strict=True)) / len(site_x)


def run(recompute_admm: bool = True) -> CoxLassoResult:
    """Run the standardize / screen / fit pipeline under encryption.

    ``recompute_admm=False`` stops after screening, which is the cheap
    part. The two ADMM runs at K = 100, unencrypted and encrypted, take
    over an hour together.
    """
    sites = _sites_raw()
    n_total = sum(len(s["time"]) for s in sites)
    p_raw = sites[0]["X"].shape[1]

    ctx = fhe_context("CKKS", **CKKS_PARAMS)
    key_holders = [make_worker(s["name"], None, lambda d, t: 0.0) for s in sites]
    master = make_threshold_master("Aggregator", ctx, key_holders)

    # Each site kept its own secret share and a copy of the public
    # parameters from that one exchange, and encrypts with them itself.

    # -- 1. pooled standardization, under encryption -----------------
    # Each site forms its own column sums and sums of squares and
    # encrypts them where they were computed; only the pooled moments
    # are ever decrypted. Encrypting at the aggregator instead would
    # mean handing it the per-site sums in the clear first, which is
    # the disclosure this round exists to avoid.
    enc_s = sum(
        h.encrypt(s["X"].sum(axis=0)) for h, s in zip(key_holders, sites, strict=True)
    )
    enc_q = sum(
        h.encrypt((s["X"] ** 2).sum(axis=0))
        for h, s in zip(key_holders, sites, strict=True)
    )
    mu = np.asarray(master.decrypt(enc_s, length=p_raw), dtype=float) / n_total
    q = np.asarray(master.decrypt(enc_q, length=p_raw), dtype=float) / n_total
    sigma = np.sqrt(np.maximum(q - mu**2, np.finfo(float).eps))

    # Cleartext reference for the same quantities, which is what makes
    # pool_agree_mu / pool_agree_sigma a measure of the CKKS error.
    s_ref = sum(s["X"].sum(axis=0) for s in sites) / n_total
    q_ref = sum((s["X"] ** 2).sum(axis=0) for s in sites) / n_total
    sigma_ref = np.sqrt(np.maximum(q_ref - s_ref**2, np.finfo(float).eps))

    sites_std = [{**s, "X": (s["X"] - mu) / sigma} for s in sites]

    # -- 2. screening, under encryption ------------------------------
    # Site side: score and information at beta = 0 on the site's own
    # rows, encrypted before either leaves.
    UI = [_score_info_at_zero(s["X"], s["time"], s["status"]) for s in sites_std]
    U = np.asarray(
        master.decrypt(
            sum(h.encrypt(u) for h, (u, _) in zip(key_holders, UI, strict=True)),
            length=p_raw,
        ),
        dtype=float,
    )
    info = np.asarray(
        master.decrypt(
            sum(h.encrypt(i) for h, (_, i) in zip(key_holders, UI, strict=True)),
            length=p_raw,
        ),
        dtype=float,
    )
    Z = U / np.sqrt(np.maximum(info, np.finfo(float).eps))
    # kind="stable" matters: probes whose scores are nearly tied at the
    # K boundary must break by index rather than arbitrarily, or the
    # retained set changes between runs. 1-based for readability.
    top_idx = np.argsort(-np.abs(Z), kind="stable")[:K] + 1

    # The same screen with the sums formed in the clear.
    U_ref = sum(u for u, _ in UI)
    info_ref = sum(i for _, i in UI)
    Z_ref = U_ref / np.sqrt(np.maximum(info_ref, np.finfo(float).eps))
    top_ref = np.argsort(-np.abs(Z_ref), kind="stable")[:K] + 1
    screen_match = set(top_idx.tolist()) == set(top_ref.tolist())

    sites_KS = [{**s, "X": s["X"][:, top_idx - 1]} for s in sites_std]

    beta_central = beta_plain = beta_admm = None
    n_iter = n_iter_plain = None
    trajectory: list[np.ndarray] = []

    if recompute_admm:
        # -- 3a. centralized lasso fit ------------------------------
        beta = cp.Variable(K)
        nll = cp.sum(
            [_breslow_nll(beta, s["X"], s["time"], s["status"]) for s in sites_KS]
        )
        prob = cp.Problem(cp.Minimize(nll + LAMBDA * cp.norm1(beta)))
        _solve(prob)
        beta_central = np.asarray(beta.value, dtype=float).ravel()

        locals_ = [build_local(s["X"], s["time"], s["status"], RHO) for s in sites_KS]

        # -- 3b. consensus ADMM with an unencrypted average ---------
        beta_plain, traj_plain = run_admm(locals_, plain_consensus)
        n_iter_plain = len(traj_plain)

        # -- 3c. the same ADMM with the encrypted average -----------
        def encrypted_consensus(site_x, site_u):
            # Each site encrypts its own x_k + u_k; the aggregator
            # receives only the encrypted terms, adds them, scales by
            # 1/N, and decrypts the average.
            ct = sum(
                h.encrypt(x + u)
                for h, x, u in zip(key_holders, site_x, site_u, strict=True)
            )
            return np.asarray(
                master.decrypt(ct * (1.0 / len(site_x)), length=K), dtype=float
            )

        beta_admm, trajectory = run_admm(locals_, encrypted_consensus)
        n_iter = len(trajectory)

    return CoxLassoResult(
        top_idx=top_idx,
        sigma=sigma,
        pool_agree_mu=float(np.max(np.abs(mu - s_ref))),
        pool_agree_sigma=float(np.max(np.abs(sigma - sigma_ref))),
        screen_match=screen_match,
        beta_centralized=beta_central,
        beta_plain=beta_plain,
        n_iter_plain=n_iter_plain,
        beta_admm=beta_admm,
        n_iter=n_iter,
        trajectory=trajectory,
        context=ctx,
        master=master,
    )


if __name__ == "__main__":  # pragma: no cover
    import sys

    full = "--screen-only" not in sys.argv
    r = run(recompute_admm=full)
    print(f"probes screened              : {r.top_idx.size} of 6416")
    print(f"encrypted pooling error  mu  : {r.pool_agree_mu:.2e}")
    print(f"                        sigma: {r.pool_agree_sigma:.2e}")
    print(f"screen matches the clear     : {r.screen_match}")
    if r.beta_admm is not None:
        print(f"ADMM iterations (plain, enc) : {r.n_iter_plain}, {r.n_iter}")
        print(f"non-zero coefficients        : {r.n_nonzero} of {K}")
        print(f"encrypted vs plain ADMM      : {r.admm_vs_plain:.3e}")
        print(f"plain ADMM vs centralized    : {r.plain_vs_centralized:.3e}")
