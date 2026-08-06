"""Differential privacy layered on the encrypted protocols.

Ports homomorpheR's ``cox-threshold-dp.Rmd`` and
``cvxr-consensus-admm-dp.Rmd``.

**This is a demonstration of what happens if you add DP, not a
recommendation.** The lossless threshold-FHE protocols are the main
story; they release the exact aggregate and leak nothing else. DP
releases a *deliberately corrupted* aggregate to bound what repeated
queries reveal, and the cost of that is the subject here.

The mechanism
-------------

Gaussian: release ``f(D) + N(0, sigma^2)``. Each of the N sites adds
``N(0, sigma^2/N)`` to its own contribution, so the sum carries
``N(0, sigma^2)`` exactly. Nobody holds the noiseless value at any
point -- an aggregator that is compromised sees ciphertexts of
*already noised* contributions.

Accounting is zCDP (Bun & Steinke 2016): a Gaussian release is
``rho = (Delta/sigma)^2 / 2``-zCDP, T queries compose to ``T*rho``,
and ``epsilon = rho + 2*sqrt(rho*log(1/delta))``. Sensitivity
``Delta = 1`` is a placeholder throughout, as in the R vignettes --
deriving a real sensitivity bound for a Cox partial likelihood is a
separate problem.

Why gradient-based optimizers fail first
----------------------------------------

A finite-difference gradient divides a difference of noisy function
values by the step ``h``, so per-coordinate gradient noise is about
``sigma*sqrt(2)/(2h)`` -- at ``h = 1e-3``, a **707x** amplification of
the function-value noise. Near the optimum the true gradient is small
and that noise dominates.

Measured here, at the DLBCL Cox fit, against the centralized estimate:

======  =========================  ==============================
sigma   BFGS (finite-diff grad)    Nelder-Mead (gradient-free)
======  =========================  ==============================
0       exact, 5/5 signs           0.002, 5/5 signs
1e-4    **collapsed to beta = 0**  0.009, 5/5 signs
1e-2    collapsed                  0.319, 4/5 signs
1e-1    collapsed                  collapsed, 2/5 signs
======  =========================  ==============================

The ordering is the vignette's point and it reproduces strongly:
gradient-free search tolerates roughly three orders of magnitude more
noise than gradient-based search, at the price of many more queries --
which costs privacy budget in turn.

**The absolute threshold differs from R, and the reason is worth
knowing.** The R vignette's BFGS stays usable to sigma = 1e-2; scipy's
collapses by 1e-5. R's ``optim`` (vmmin) accepts a step on a
*function-value* decrease alone, whereas scipy's BFGS uses a Wolfe
line search whose curvature condition evaluates the *gradient* at each
trial point -- so the 1/h amplification hits the line search itself,
not merely the search direction. scipy is therefore strictly more
fragile here, which if anything strengthens the vignette's claim that
any difference-based optimizer meets this failure mode.

Both are properties of the mechanism meeting the optimizer, not
defects in the protocol: the protocol releases exactly what it
promises.

On randomness
-------------

The noise is drawn with numpy, not replayed from R. Two independent DP
runs never agree on values *within* a language either, so a
value-level cross-language comparison would be meaningless. What is
comparable, and what the tests assert, is the behaviour: sigma = 0
reproduces the lossless fit, error grows with sigma, and gradient-based
search collapses before gradient-free search does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from homomorphepy.actors import ThresholdSite, make_threshold_master
from homomorphepy.context import Context, fhe_context
from homomorphepy.examples.cox import CKKS_PARAMS, COVARIATES, local_cox_nll
from homomorphepy.examples.cox import _site_frames as _cox_sites

__all__ = [
    "DEFAULT_DELTA",
    "SENSITIVITY",
    "DPFit",
    "amplification_factor",
    "budget",
    "compare_optimizers",
    "fit_at_sigma",
    "sweep",
    "zcdp_to_epsilon",
]

DEFAULT_DELTA = 1e-5
SENSITIVITY = 1.0  # placeholder, as in the R vignettes
FINITE_DIFF_STEP = 1e-3


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

    ``sigma*sqrt(2)/(2h)``: at h = 1e-3 this is ~707, the figure the R
    vignette quotes.
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
    workers = [ThresholdSite(n, sites[n], noisy_local) for n in names]
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
    # returns x0 unchanged and still reports success. R's optim uses
    # ndeps = 1e-3, giving the ~707x quoted in the vignette; matching
    # it is what makes the two languages tell the same story.
    if method == "BFGS":
        options = {"gtol": 1e-4, "finite_diff_rel_step": FINITE_DIFF_STEP}
    elif method == "L-BFGS-B":
        options = {"ftol": 1e-9, "finite_diff_rel_step": FINITE_DIFF_STEP}
    else:
        options = {"xatol": 1e-4, "fatol": 1e-4, "maxfev": 4000}
    fit = minimize(
        objective, x0=np.zeros(len(COVARIATES)), method=method, options=options
    )
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
    sigmas=(0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    method: str = "Nelder-Mead",
    seed: int = 1,
) -> list[DPFit]:
    """Fit across four orders of magnitude of noise."""
    return [fit_at_sigma(s, method=method, seed=seed) for s in sigmas]


def compare_optimizers(sigma: float = 1e-4, seed: int = 1) -> dict[str, DPFit]:
    """Gradient-based vs gradient-free search at the same noise scale."""
    return {
        "BFGS": fit_at_sigma(sigma, method="BFGS", seed=seed),
        "Nelder-Mead": fit_at_sigma(sigma, method="Nelder-Mead", seed=seed),
    }


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
