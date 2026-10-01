"""Differential privacy layered on the encrypted protocols.

**This is a demonstration of what happens if you add DP, not a
recommendation.** The threshold-FHE protocols without noise are the
main path; what they reveal is the exact aggregate at each query. DP
releases a *deliberately perturbed* aggregate to bound what repeated
queries reveal, and the cost of that is the subject here.

The mechanism
-------------

Gaussian: release ``f(D) + N(0, sigma^2)``. Each of the N sites adds
``N(0, sigma^2/N)`` to its own contribution, so the sum carries
``N(0, sigma^2)`` exactly. Nobody holds the noiseless value at any
point -- an aggregator that is compromised sees encrypted
*already noised* contributions.

Accounting is zCDP (Bun & Steinke 2016): a Gaussian release is
``rho = (Delta/sigma)^2 / 2``-zCDP, T queries compose to ``T*rho``,
and ``epsilon = rho + 2*sqrt(rho*log(1/delta))``. Sensitivity
``Delta = 1`` is a placeholder throughout: deriving a defensible
sensitivity bound for a Cox partial likelihood is a separate problem
and is not attempted here.

The sweep
---------

BFGS and Nelder-Mead fits at each sigma, against the centralized
estimate, are recorded by ``docs/_recorded/record_dp_sweep.py`` and
tabulated in ``docs/dp.qmd``. BFGS estimates the gradient by finite
differences of the noisy objective.

On randomness
-------------

Two DP runs never agree on values, by construction -- the mechanism is
randomized, so a specific coefficient is not a reproducible quantity
and nothing here should be compared value-for-value against anything.
What is stable, and what the tests assert, is the behavior: sigma = 0
reproduces the lossless fit, error grows with sigma, and gradient-based
search collapses before gradient-free search does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from homomorphepy.actors import make_threshold_master, make_worker
from homomorphepy.context import Context, fhe_context
from homomorphepy.examples import _consensus
from homomorphepy.examples._consensus import (
    ConsensusSite,
    admm_loop,
    centralized,
    plain_consensus,
)
from homomorphepy.examples.cox import CKKS_PARAMS, COVARIATES, local_cox_nll
from homomorphepy.examples.cox import _site_frames as _cox_sites

__all__ = [
    "CONSENSUS_DGP",
    "DEFAULT_DELTA",
    "SENSITIVITY",
    "ConsensusSite",
    "DPFit",
    "DPConsensusFit",
    "DPConsensusResult",
    "RhoChoice",
    "admm_loop",
    "amplification_factor",
    "budget",
    "centralized",
    "choose_rho_and_T",
    "compare_optimizers",
    "consensus_at_sigma",
    "consensus_sweep",
    "fit_at_sigma",
    "plain_consensus",
    "simulate_cohort",
    "sweep",
    "zcdp_to_epsilon",
]

DEFAULT_DELTA = 1e-5
SENSITIVITY = 1.0  # placeholder; see the module docstring
FINITE_DIFF_STEP = 1e-3
NM_SIMPLEX_STEP = 0.1
NM_RELTOL = 1e-7
NM_MAXFEV = 500


def zcdp_to_epsilon(rho: float, delta: float = DEFAULT_DELTA) -> float:
    """Convert accumulated zCDP ``rho`` to an ``(epsilon, delta)`` pair."""
    return float(rho + 2 * math.sqrt(rho * math.log(1 / delta)))


def budget(
    n_queries: int,
    sigma: float,
    delta: float = DEFAULT_DELTA,
    sensitivity: float = SENSITIVITY,
) -> dict[str, float]:
    """zCDP composition for ``n_queries`` Gaussian releases."""
    if sigma <= 0:
        return {
            "n_queries": n_queries,
            "sigma": sigma,
            "rho_per_query": math.inf,
            "rho_total": math.inf,
            "epsilon": math.inf,
        }
    rho_q = (sensitivity / sigma) ** 2 / 2
    rho_total = n_queries * rho_q
    return {
        "n_queries": n_queries,
        "sigma": sigma,
        "rho_per_query": rho_q,
        "rho_total": rho_total,
        "epsilon": zcdp_to_epsilon(rho_total, delta),
    }


def amplification_factor(h: float = FINITE_DIFF_STEP) -> float:
    """How much a central-difference gradient inflates function noise.

    ``sqrt(2)/(2h)``: at h = 1e-3 this is ~707.
    """
    return math.sqrt(2) / (2 * h)


@dataclass
class DPFit:
    sigma: float
    method: str
    coefficients: dict[str, float]
    centralized: dict[str, float]
    max_abs_diff: float
    sign_agreement: int
    n_queries: int
    converged: bool
    epsilon: float
    context: Context = field(repr=False)


def fit_at_sigma(
    sigma: float,
    method: str = "Nelder-Mead",
    seed: int = 1,
    delta: float = DEFAULT_DELTA,
) -> DPFit:
    """Fit the threshold-Cox model with per-site Gaussian noise.

    Each site adds ``N(0, sigma^2/N)`` to its local negative
    log-likelihood before encryption, so the decrypted aggregate
    carries ``N(0, sigma^2)``.
    """
    sites = _cox_sites()
    names = list(sites)
    n_sites = len(names)
    rng = np.random.default_rng(seed)
    scale = sigma / math.sqrt(n_sites)

    def noisy_local(data, beta):
        value = local_cox_nll(data, beta)
        if value is None or math.isnan(value):
            return math.nan
        return value + (rng.normal(0.0, scale) if sigma > 0 else 0.0)

    ctx = fhe_context("CKKS", **CKKS_PARAMS)
    workers = [make_worker(n, sites[n], noisy_local) for n in names]
    master = make_threshold_master("Aggregator", ctx, workers)

    calls = {"n": 0}

    def objective(beta):
        calls["n"] += 1
        v = master.aggregate(np.asarray(beta, dtype=float))
        return 1e12 if (v is None or math.isnan(v)) else float(v)

    # The finite-difference step is LOAD-BEARING here, unlike in
    # examples/cox.py where sweeping it changed nothing. Without DP
    # noise the encrypted objective is accurate to ~1e-13 and any step
    # works. Under DP the gradient noise is the function noise divided
    # by h, so scipy's default (~1.5e-8) amplifies sigma by ~7e7 and
    # destroys the gradient even at sigma = 1e-4 -- the optimizer then
    # returns x0 unchanged and still reports success. At h = 1e-3 the
    # amplification is ~707x, which is survivable; the constant is
    # load-bearing rather than decorative here.
    x0 = np.zeros(len(COVARIATES))
    fun = objective
    if method == "BFGS":
        options = {"gtol": 1e-4, "finite_diff_rel_step": FINITE_DIFF_STEP}
    elif method == "L-BFGS-B":
        options = {"ftol": 1e-9, "finite_diff_rel_step": FINITE_DIFF_STEP}
    elif method == "Nelder-Mead":
        # Starting simplex: NM_SIMPLEX_STEP added to one coordinate per
        # vertex. Stop when the spread of objective values over the
        # simplex is at most NM_RELTOL * (|f(x0)| + NM_RELTOL), or after
        # NM_MAXFEV evaluations. f(x0) is evaluated once and reused as
        # the first simplex vertex, so it costs one query, not two.
        f0 = objective(x0)
        first = {"pending": True}

        def fun(beta):
            if first["pending"] and np.array_equal(beta, x0):
                first["pending"] = False
                return f0
            return objective(beta)

        options = {
            "initial_simplex": np.vstack([x0, x0 + NM_SIMPLEX_STEP * np.eye(len(x0))]),
            "xatol": np.inf,
            "fatol": NM_RELTOL * (abs(f0) + NM_RELTOL),
            "maxfev": NM_MAXFEV,
        }
    else:
        raise ValueError(f"unsupported method {method!r}")
    fit = minimize(fun, x0=x0, method=method, options=options)
    beta_hat = np.asarray(fit.x, dtype=float)

    # Centralized cleartext fit: what DP is degrading away from.
    from statsmodels.duration.hazard_regression import PHReg

    from homomorphepy.fixtures import load_dlbcl

    df = load_dlbcl()
    ref = PHReg(
        df["time"].to_numpy(dtype=float),
        np.column_stack([df[c].to_numpy(dtype=float) for c in COVARIATES]),
        status=df["status"].to_numpy(dtype=int),
        strata=df["Subgroup"].cat.codes.to_numpy(),
        ties="efron",
    ).fit()
    beta_ref = np.asarray(ref.params, dtype=float)

    return DPFit(
        sigma=sigma,
        method=method,
        coefficients=dict(zip(COVARIATES, beta_hat.tolist(), strict=True)),
        centralized=dict(zip(COVARIATES, beta_ref.tolist(), strict=True)),
        max_abs_diff=float(np.max(np.abs(beta_hat - beta_ref))),
        # How many coefficients still land on the correct side of zero:
        # the qualitative conclusion, which survives longer than the
        # point estimates do.
        sign_agreement=int(np.sum(np.sign(beta_hat) == np.sign(beta_ref))),
        n_queries=calls["n"],
        converged=bool(fit.success),
        epsilon=budget(calls["n"], sigma, delta)["epsilon"],
        context=ctx,
    )


def sweep(
    sigmas=(0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    method: str = "Nelder-Mead",
    seed: int = 1,
) -> list[DPFit]:
    """Fit at zero noise and across five orders of magnitude of noise."""
    return [fit_at_sigma(s, method=method, seed=seed) for s in sigmas]


def compare_optimizers(sigma: float = 1e-4, seed: int = 1) -> dict[str, DPFit]:
    """Gradient-based vs gradient-free search at the same noise scale."""
    return {
        "BFGS": fit_at_sigma(sigma, method="BFGS", seed=seed),
        "Nelder-Mead": fit_at_sigma(sigma, method="Nelder-Mead", seed=seed),
    }


# ---------------------------------------------------------------------
# The same mechanism on consensus ADMM
# ---------------------------------------------------------------------


# L2-regularized logistic regression across three sites. The cohort
# and the surrogate share the design -- site sizes and covariate
# schema -- and differ in the effect sizes: the cohort's are the truth,
# the surrogate's are nominal values fixed in advance.
CONSENSUS_DGP = dict(
    covariates=("intercept", "age", "bmi", "sex"),
    beta_true=(-0.5, 0.4, -0.3, 0.6),
    beta_nominal=(0.0, 0.5, 0.5, 0.5),
    n_per=(500, 1000, 1500),
    lam=1.0,
    tol=1e-3,
    max_iter=60,
    rho_grid=(10.0, 20.0, 50.0, 100.0, 500.0),
    sigmas=(0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    seed=20260412,
    surrogate_seed=20260413,
)


def simulate_cohort(beta: Sequence[float], seed: int):
    """Three sites of ``(X, y)``: an intercept, two normal covariates, a
    Bernoulli(0.5) covariate, and a logistic outcome."""
    rng = np.random.default_rng(seed)
    beta = np.asarray(beta, dtype=float)
    cohort = []
    for n in CONSENSUS_DGP["n_per"]:
        X = np.column_stack(
            [
                np.ones(n),
                rng.normal(size=n),
                rng.normal(size=n),
                rng.binomial(1, 0.5, size=n),
            ]
        ).astype(float)
        prob = 1.0 / (1.0 + np.exp(-(X @ beta)))
        cohort.append((X, (rng.uniform(size=n) < prob).astype(float)))
    return cohort


def _build(cohort, rho: float) -> list[ConsensusSite]:
    lam, N = CONSENSUS_DGP["lam"], len(cohort)
    return [
        ConsensusSite(f"Site {i + 1}", X, y, rho, lam, N)
        for i, (X, y) in enumerate(cohort)
    ]


@dataclass
class RhoChoice:
    rho: float
    T: int
    sweep: list[dict]


def choose_rho_and_T(surrogate) -> RhoChoice:
    """Pick ``rho`` and the iteration count ``T`` on the surrogate.

    Runs noiseless ADMM in the clear at each ``rho`` in the grid and
    keeps the one that converges in the fewest iterations; ``T`` is that
    count. The surrogate carries no record from any site, so the choice
    is not a release, and it needs no encryption.
    """
    p = surrogate[0][0].shape[1]
    tol, max_iter = CONSENSUS_DGP["tol"], CONSENSUS_DGP["max_iter"]
    rows = []
    for rho in CONSENSUS_DGP["rho_grid"]:
        _, k, _, ok = admm_loop(
            _build(surrogate, rho), p, rho, max_iter, tol, plain_consensus
        )
        rows.append({"rho": float(rho), "iters": k, "converged": ok})
    converged = [r for r in rows if r["converged"]]
    if not converged:
        raise RuntimeError("no rho in the grid converged within max_iter")
    # First minimum in grid order, as a tie would be broken by hand.
    best = min(converged, key=lambda r: r["iters"])
    return RhoChoice(rho=best["rho"], T=best["iters"], sweep=rows)


@dataclass
class DPConsensusFit:
    sigma: float
    beta: np.ndarray
    max_dev: float
    trajectory: list[np.ndarray] = field(repr=False)


def consensus_at_sigma(
    sigma: float,
    cohort,
    rho: float,
    T: int,
    beta_centralized: np.ndarray,
    seed: int | None = None,
) -> DPConsensusFit:
    """Run ``T`` iterations of consensus ADMM with output-DP noise.

    Each site adds ``N(0, sigma^2 * N)`` to its ``x + u`` vector and
    encrypts the result, all before anything leaves the site. The
    encrypted draws sum under the joint key and the ``1/N`` scaling
    contracts the variance back to ``sigma^2`` per coordinate, so the
    released consensus carries ``N(0, sigma^2)`` and no party other than
    the site ever holds its noiseless vector.

    The loop runs exactly ``T`` iterations (``tol = 0``): residuals
    cannot shrink below the noise floor, and a data-dependent stopping
    time would be a release the budget does not count.
    """
    sites = _build(cohort, rho)
    N, p = len(sites), cohort[0][0].shape[1]
    ctx = fhe_context("CKKS", **_consensus.CKKS_PARAMS)
    master = make_threshold_master("Aggregator", ctx, sites)
    # One generator per site: each site draws its own noise.
    rngs = [np.random.default_rng([seed or 0, i]) for i in range(N)]

    def site_contribution_dp(site, rng):
        noise = rng.normal(0.0, sigma * math.sqrt(N), size=p) if sigma > 0 else 0.0
        return site.encrypt(site.x_curr + site.u_curr + noise)

    def encrypted_consensus_dp(sites):
        cts = [site_contribution_dp(s, r) for s, r in zip(sites, rngs, strict=True)]
        ct_avg = sum(cts) * (1.0 / N)
        return np.asarray(master.decrypt(ct_avg, length=p), dtype=float)

    z, _, trajectory, _ = admm_loop(sites, p, rho, T, 0.0, encrypted_consensus_dp)
    return DPConsensusFit(
        sigma=float(sigma),
        beta=z,
        max_dev=float(np.max(np.abs(z - beta_centralized))),
        trajectory=trajectory,
    )


@dataclass
class DPConsensusResult:
    choice: RhoChoice
    beta_centralized: np.ndarray
    fits: list[DPConsensusFit]
    clean_dev: float
    tol: float


def consensus_sweep(
    sigmas: Sequence[float] | None = None,
    cohort=None,
    surrogate=None,
) -> DPConsensusResult:
    """The whole example: choose ``rho`` and ``T``, then sweep ``sigma``.

    Simulates the cohort and the surrogate unless they are passed in as
    sequences of ``(X, y)`` pairs, one per site. Raises if the
    ``sigma = 0`` run is not within ``10 * tol`` of the centralized
    fit: with the noise off, the fixed-``T`` protocol must reach it, or
    the other rows say nothing about privacy.
    """
    d = CONSENSUS_DGP
    sigmas = tuple(d["sigmas"] if sigmas is None else sigmas)
    if cohort is None:
        cohort = simulate_cohort(d["beta_true"], d["seed"])
    if surrogate is None:
        surrogate = simulate_cohort(d["beta_nominal"], d["surrogate_seed"])

    choice = choose_rho_and_T(surrogate)
    beta_central = centralized(cohort, d["lam"])
    fits = [
        consensus_at_sigma(s, cohort, choice.rho, choice.T, beta_central, seed=100 + j)
        for j, s in enumerate(sigmas, start=1)
    ]
    clean = [f for f in fits if f.sigma == 0.0]
    clean_dev = clean[0].max_dev if clean else math.nan
    if clean and clean_dev > 10 * d["tol"]:
        raise RuntimeError(
            f"at sigma = 0 the protocol is {clean_dev:.2e} from the "
            f"centralized fit, more than 10 * tol = {10 * d['tol']:g}"
        )
    return DPConsensusResult(
        choice=choice,
        beta_centralized=beta_central,
        fits=fits,
        clean_dev=clean_dev,
        tol=d["tol"],
    )


if __name__ == "__main__":  # pragma: no cover
    print(
        f"finite-difference noise amplification at h={FINITE_DIFF_STEP:g}: "
        f"{amplification_factor():.0f}x\n"
    )

    print("sigma sweep (Nelder-Mead), against the centralized cleartext fit")
    print(f"{'sigma':>8}{'max|diff|':>12}{'signs ok':>10}{'queries':>9}{'epsilon':>14}")
    for f in sweep():
        print(
            f"{f.sigma:>8g}{f.max_abs_diff:>12.4f}{f.sign_agreement:>7}/5"
            f"{f.n_queries:>9}{f.epsilon:>14.3g}"
        )

    print("\ngradient-based vs gradient-free at sigma = 1e-4")
    both = compare_optimizers(1e-4)
    print(
        f"{'method':>14}{'max|diff|':>12}{'signs ok':>10}{'queries':>9}{'epsilon':>14}"
    )
    for name, f in both.items():
        print(
            f"{name:>14}{f.max_abs_diff:>12.4f}{f.sign_agreement:>7}/5"
            f"{f.n_queries:>9}{f.epsilon:>14.3g}"
        )
