"""Cox-lasso on DLBCL gene expression, federated under threshold FHE.

The largest example in the set, and the only one where every stage of
the pipeline needs the encrypted channel:

1. **Standardization.** Column means and second moments are pooled
   across sites under encryption, so no site's marginal distribution
   is revealed.
2. **Screening.** 6416 probes are ranked by a univariate Cox score at
   beta = 0, again pooled under encryption, and the top K = 100 kept.
   Fitting all 6416 exhausts memory during canonicalization.
3. **Consensus ADMM.** The lasso-penalized stratified Cox fit is
   reached by ADMM, with only the consensus average traversing the
   encrypted channel.

Two things differ from :mod:`.consensus_admm` beyond scale.

**The tie convention is Breslow here, not Efron.** The partial
likelihood is built symbolically as ``log_sum_exp(eta[i:]) - eta[i]``
over event times so cvxpy can canonicalize it; that expression *is*
the Breslow form. Efron has no comparable convex-atom formulation.
So this example and :mod:`.cox` deliberately use different
conventions -- Efron there via PHReg, Breslow here by construction.

**Measured data, so it ships with the package.** Unlike the simulated
examples there is no draw to repeat: the DLBCL cohort is what it is,
and every run reads the same bytes.

Expensive: the ADMM runs to ~150 iterations with three per-site conic
solves each. ``run(recompute_admm=False)`` performs the standardization
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
from homomorphepy.examples.consensus_admm import SOLVER
from homomorphepy.fixtures import load_dlbcl, load_dlbcl_gex, site_order

__all__ = ["CoxLassoResult", "run", "K", "LAMBDA", "RHO"]

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


@dataclass
class CoxLassoResult:
    top_idx: np.ndarray
    sigma: np.ndarray
    pool_agree_mu: float
    pool_agree_sigma: float
    beta_centralized: np.ndarray | None
    beta_admm: np.ndarray | None
    n_iter: int | None
    # The consensus iterate after every ADMM sweep, so the run can be
    # plotted as trajectories rather than only as its endpoint. One
    # row per iteration, K columns; empty when the ADMM was skipped.
    trajectory: list[np.ndarray]
    context: Context = field(repr=False)
    master: ThresholdMaster = field(repr=False)

    @property
    def n_nonzero(self) -> int | None:
        """How many of the K screened probes the lasso retained."""
        if self.beta_admm is None:
            return None
        return int(np.sum(np.abs(self.beta_admm) > 1e-8))

    @property
    def admm_vs_centralized(self) -> float | None:
        """Max abs difference: the ADMM fit against the centralized one.

        The split-vs-pooled comparison. Bounded by the ADMM stopping
        tolerance, not by CKKS noise -- see :attr:`pool_agree_mu` for
        the size of the encryption error itself.
        """
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

    ``np.lexsort((-status, time))`` orders by time, breaking ties so
    events precede censorings -- note that lexsort takes its keys in
    reverse priority order, one of the two sort traps that would
    silently change which probes are kept.
    """
    order = np.lexsort((-status, time))
    Xo, so = X[order], status[order]
    n, p = Xo.shape
    U = np.zeros(p)
    info = np.zeros(p)
    for i in range(n):
        if so[i] == 1:
            risk = Xo[i:]
            mu = risk.mean(axis=0)
            U += Xo[i] - mu
            info += ((risk - mu) ** 2).sum(axis=0) / risk.shape[0]
    return U, info


def _breslow_nll(beta, X, time, status):
    """Cox partial NLL as a cvxpy expression (Breslow ties).

    ``sum_j in events [ log_sum_exp(eta_{j:}) - eta_j ]`` over
    event-time-ordered rows. This is exactly the Breslow partial
    likelihood; Efron has no equivalent convex-atom form, which is why
    this example and :mod:`.cox` use different tie conventions.
    """
    order = np.lexsort((-status, time))
    Xo, so = X[order], status[order]
    eta = Xo @ beta
    terms = [cp.log_sum_exp(eta[i:]) - eta[i] for i in range(len(so)) if so[i] == 1]
    return cp.sum(terms) if len(terms) > 1 else terms[0]


def _solve(problem):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        problem.solve(solver=SOLVER, verbose=False)


def run(recompute_admm: bool = True) -> CoxLassoResult:
    """Run the standardize / screen / fit pipeline under encryption.

    ``recompute_admm=False`` stops after screening, which is the cheap
    part; the consensus ADMM at K = 100 takes about half an hour.
    """
    sites = _sites_raw()
    n_total = sum(len(s["time"]) for s in sites)
    p_raw = sites[0]["X"].shape[1]

    ctx = fhe_context("CKKS", **CKKS_PARAMS)
    key_holders = [make_worker(s["name"], None, lambda d, t: 0.0) for s in sites]
    master = make_threshold_master("Aggregator", ctx, key_holders)

    # What each site kept from that one exchange: its own secret share
    # and a copy of the public parameters — the crypto context and the
    # joint public key, and no share of anyone else's. Asking a site
    # what it holds involves no aggregator, and it is all a site needs
    # in order to encrypt.
    pub = [h.params for h in key_holders]

    # -- 1. pooled standardization, under encryption -----------------
    # Each site forms its own column sums and sums of squares and
    # encrypts them where they were computed; only the pooled moments
    # are ever decrypted. Encrypting at the aggregator instead would
    # mean handing it the per-site sums in the clear first, which is
    # the disclosure this round exists to avoid.
    enc_s = sum(p.encrypt(s["X"].sum(axis=0)) for p, s in zip(pub, sites, strict=True))
    enc_q = sum(
        p.encrypt((s["X"] ** 2).sum(axis=0)) for p, s in zip(pub, sites, strict=True)
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
            sum(p.encrypt(u) for p, (u, _) in zip(pub, UI, strict=True)),
            length=p_raw,
        ),
        dtype=float,
    )
    info = np.asarray(
        master.decrypt(
            sum(p.encrypt(i) for p, (_, i) in zip(pub, UI, strict=True)),
            length=p_raw,
        ),
        dtype=float,
    )
    Z = U / np.sqrt(np.maximum(info, np.finfo(float).eps))
    # kind="stable" matters: probes whose scores are nearly tied at the
    # K boundary must break by index rather than arbitrarily, or the
    # retained set changes between runs. 1-based for readability.
    top_idx = np.argsort(-np.abs(Z), kind="stable")[:K] + 1

    sites_KS = [{**s, "X": s["X"][:, top_idx - 1]} for s in sites_std]

    beta_central = beta_admm = None
    n_iter = None
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

        # -- 3b. consensus ADMM with encrypted averaging ------------
        locals_ = []
        for s in sites_KS:
            x = cp.Variable(K)
            zp = cp.Parameter(K, value=np.zeros(K))
            up = cp.Parameter(K, value=np.zeros(K))
            obj = _breslow_nll(x, s["X"], s["time"], s["status"]) + (RHO / 2) * (
                cp.sum_squares(x - zp + up)
            )
            locals_.append(
                {"prob": cp.Problem(cp.Minimize(obj)), "x": x, "zp": zp, "up": up}
            )

        site_x = [np.zeros(K) for _ in sites_KS]
        site_u = [np.zeros(K) for _ in sites_KS]
        z_curr = np.zeros(K)
        n_sites = len(sites_KS)

        n_iter = 0
        for it in range(1, MAX_ITER + 1):
            n_iter = it
            for i, lp in enumerate(locals_):
                lp["zp"].value = z_curr
                lp["up"].value = site_u[i]
                _solve(lp["prob"])
                site_x[i] = np.asarray(lp["x"].value, dtype=float).ravel()

            # The only step that leaves a site, and the site encrypts
            # it before it does. The per-site x_k + u_k never exists in
            # the clear anywhere but at the site that formed it.
            ct = sum(
                p.encrypt(x + u) for p, x, u in zip(pub, site_x, site_u, strict=True)
            )
            w_avg = np.asarray(
                master.decrypt(ct * (1.0 / n_sites), length=K), dtype=float
            )
            tau = LAMBDA / (n_sites * RHO)
            z_new = np.sign(w_avg) * np.maximum(np.abs(w_avg) - tau, 0.0)

            site_u = [u + (x - z_new) for u, x in zip(site_u, site_x, strict=True)]
            primal = float(np.sqrt(np.mean([np.sum((x - z_new) ** 2) for x in site_x])))
            dual = float(RHO * np.linalg.norm(z_new - z_curr))
            z_curr = z_new
            trajectory.append(z_curr.copy())
            if primal < TOL and dual < TOL:
                break
        beta_admm = z_curr

    return CoxLassoResult(
        top_idx=top_idx,
        sigma=sigma,
        pool_agree_mu=float(np.max(np.abs(mu - s_ref))),
        pool_agree_sigma=float(np.max(np.abs(sigma - sigma_ref))),
        beta_centralized=beta_central,
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
    if r.beta_admm is not None:
        print(f"ADMM iterations              : {r.n_iter}")
        print(f"non-zero coefficients        : {r.n_nonzero} of {K}")
        print(f"ADMM vs centralized lasso    : {r.admm_vs_centralized:.3e}")
