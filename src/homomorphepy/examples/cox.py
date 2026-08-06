"""Distributed Cox regression through an encrypted channel.

Ports homomorpheR's ``cox.Rmd`` and ``cox-threshold.Rmd``. Both live
here because their whole point is that the optimizer-facing code is
*identical*: only the master class changes, from a single-decrypter
CKKS master to an n-of-n threshold one. :func:`run` takes ``backend``
and the rest of the protocol is untouched.

The setting is the Rosenwald DLBCL cohort, with the molecular subgroup
used as the site boundary -- GCB, ABC and Type III arise from different
cells of origin and tend to be diagnosed at different referral centers,
so the split is operationally realistic rather than arbitrary. The
three sites are imbalanced (115 / 71 / 49), which is also realistic.

Each site computes its own Cox partial log-likelihood at the current
coefficient vector and never shares patient rows. The site-level
function is ``PHReg(ties='efron').loglike(beta)``, which reproduces
R's ``coxph(init=beta, iter.max=0)$loglik[1]`` to ~1e-12 -- verified
independently in ``tests/test_cox_loglik.py`` before this module was
written.

``ties='efron'`` is not optional. statsmodels defaults to Breslow,
``survival`` defaults to Efron, and on this cohort the difference
reaches 3.56 in pooled log-likelihood: a silent wrong answer rather
than an error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize
from statsmodels.duration.hazard_regression import PHReg

from homomorphepy.actors import (
    Master,
    ThresholdSite,
    make_ckks_master,
    make_threshold_master,
    make_worker,
)
from homomorphepy.context import Context, fhe_context
from homomorphepy.fixtures import load_dlbcl, site_order

__all__ = ["COVARIATES", "FTOL", "CoxResult", "local_cox_nll", "run"]

COVARIATES = ["GCB_sig", "LN_sig", "Prolif_sig", "BMP6", "MHC2_sig"]

# Matches the R vignettes: the summed stratified Cox nLL on this cohort
# has magnitude ~5e2, so scaling_mod_size is lifted from 50 to 59 with
# first_mod_size 60 for margin.
CKKS_PARAMS = dict(
    multiplicative_depth=1, scaling_mod_size=59, first_mod_size=60, batch_size=8
)

# No finite-difference step is set. R's optim defaults to ndeps = 1e-3
# and the vignettes rely on it, but sweeping 1e-3 / 1e-5 / scipy's
# default here changed neither the iterate count nor the result by a
# single digit -- L-BFGS-B's path is insensitive to it on this problem.
# Carrying a constant that demonstrably does nothing would just be one
# more thing to explain, so it is gone. (Separately measured: the
# encrypted objective is accurate to ~1e-13, so the step could be
# shrunk freely if it ever did matter -- the CKKS noise floor is not
# the binding constraint here. See D6.)

# The convergence criterion, and why it is L-BFGS-B rather than BFGS.
#
# The R vignettes pass control = list(reltol = 1e-7): a RELATIVE
# FUNCTION-VALUE test -- stop when a step cannot improve the objective
# by a relative amount. scipy's `BFGS` has no such option; it tests the
# GRADIENT NORM (gtol), a different question entirely, so R's 1e-7
# carries no meaning there.
#
# `L-BFGS-B`'s ftol is
#     (f_k - f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) <= ftol
# which is R's reltol in form as well as spirit, so the vignettes'
# 1e-7 transfers directly and "converged" means the same thing in both
# languages without any calibration.
#
# The alternative was to keep `BFGS` and back-calculate an equivalent
# gtol from R's observed behaviour. That was measured
# (temp/cox_gtol_calibrate.R) and it works: R's optim converges in 36
# evaluations stopping at |grad|_inf = 1.54e-04, the central-difference
# truncation error at ndeps = 1e-3 puts an attainable floor near
# 7.6e-06, and gtol = 1e-4 sits between them -- reaching objective
# 495.229021767 against R's 495.229021767. It was rejected anyway: that
# constant is calibrated to THIS cohort at THIS objective scale (~495),
# so it would silently need re-deriving for any other dataset.
# homomorphepy is meant to be the shared infrastructure both languages
# compute on, not a pixel-match of one example, and a relative
# criterion is scale-free where a gradient threshold is not.
#
# Worth recording from that investigation: an earlier gtol of 1e-6 lay
# BELOW the attainable floor and so reported success=False -- on the
# CLEARTEXT objective too. The failure was never about encryption.
#
# Matching the criterion's FORM does not make two implementations stop
# in the same PLACE -- vmmin and L-BFGS-B take different steps -- so
# R's 1e-7 is not the right value here even though it is the right
# kind of tolerance. Swept against the centralized fit:
#
#     ftol       max|b - b_centralized|   in SE units   evals
#     1e-7  (R)          1.32e-04            1.1e-03      54
#     2.2e-9 (scipy)     1.29e-05            1.1e-04      60
#     1e-9               1.77e-07            1.7e-06      66
#     1e-12              1.90e-07            1.3e-06      72
#     (R's own optim)    1.82e-06            2.1e-05      36
#
# 1e-9 reaches the accuracy floor -- 1e-12 buys nothing, so the limit
# is the finite-difference gradient rather than the tolerance -- and
# lands an order of magnitude tighter than R for six extra evaluations.
#
# This is a relative criterion driven to the numerical floor, not a
# constant fitted to this cohort: it carries no dependence on the
# objective's scale, which is exactly why it was preferred over
# back-calculating a gradient threshold from R's stopping point.
FTOL = 1e-9

PENALTY = 1e12


def _site_frames() -> dict[str, dict[str, np.ndarray]]:
    """Split the cohort by subgroup, in the protocol's site order."""
    df = load_dlbcl()
    out = {}
    for name in site_order():
        m = (df["Subgroup"] == name).to_numpy()
        out[name] = {
            "time": df["time"].to_numpy(dtype=float)[m],
            "status": df["status"].to_numpy(dtype=int)[m],
            "X": np.column_stack([df[c].to_numpy(dtype=float)[m] for c in COVARIATES]),
        }
    return out


def local_cox_nll(data: dict[str, np.ndarray], beta: Any) -> float:
    """One site's negative Cox partial log-likelihood at ``beta``.

    The direct analogue of the R vignettes' ``local_cox_nll``: plain
    statistical code that never touches a ciphertext. Returns NaN when
    the local computation fails, which the master propagates as a
    non-evaluable parameter.
    """
    b = np.asarray(beta, dtype=float).ravel()
    try:
        model = PHReg(data["time"], data["X"], status=data["status"], ties="efron")
        return float(-model.loglike(b))
    except Exception:
        return math.nan


@dataclass
class CoxResult:
    backend: str
    coefficients: dict[str, float]
    centralized: dict[str, float]
    max_abs_difference: float
    loglik_encrypted: float
    loglik_centralized: float
    objective_at_fit: float
    n_objective_calls: int
    converged: bool
    site_sizes: dict[str, int]
    site_events: dict[str, int]
    master: Master = field(repr=False)
    context: Context = field(repr=False)

    def table(self) -> list[dict[str, Any]]:
        """Side-by-side comparison, as the R vignettes tabulate it."""
        return [
            {
                "coefficient": name,
                "distributed_encrypted": self.coefficients[name],
                "aggregated_cleartext": self.centralized[name],
                "abs_diff": abs(self.coefficients[name] - self.centralized[name]),
            }
            for name in COVARIATES
        ]


def run(
    backend: Literal["ckks", "threshold"] = "threshold",
    start: list[float] | None = None,
) -> CoxResult:
    """Fit the stratified Cox model across sites through encryption.

    ``backend='ckks'`` uses a single-decrypter master (``cox.Rmd``);
    ``backend='threshold'`` uses n-of-n threshold keys where no party
    can decrypt alone (``cox-threshold.Rmd``). Everything after the
    master is constructed is identical between the two.
    """
    sites = _site_frames()
    names = list(sites)

    ctx = fhe_context("CKKS", **CKKS_PARAMS)

    if backend == "ckks":
        workers = [make_worker(n, sites[n], local_cox_nll) for n in names]
        master: Master = make_ckks_master("Master", ctx, ctx.KeyGen())
        master.set_workers(workers)
    elif backend == "threshold":
        workers = [ThresholdSite(n, sites[n], local_cox_nll) for n in names]
        master = make_threshold_master("Aggregator", ctx, workers)
    else:
        raise ValueError(f"unknown backend {backend!r}; use 'ckks' or 'threshold'")

    calls = {"n": 0}

    def encrypted_nll(beta):
        calls["n"] += 1
        value = master.aggregate(np.asarray(beta, dtype=float))
        return PENALTY if (value is None or math.isnan(value)) else float(value)

    fit = minimize(
        encrypted_nll,
        x0=np.zeros(len(COVARIATES)) if start is None else np.asarray(start),
        method="L-BFGS-B",
        options={"ftol": FTOL},
    )
    beta_hat = np.asarray(fit.x, dtype=float)

    # Centralized reference: one stratified fit on the pooled cohort,
    # which is what the distributed protocol is meant to reproduce.
    df = load_dlbcl()
    strata = df["Subgroup"].cat.codes.to_numpy()
    centralized = PHReg(
        df["time"].to_numpy(dtype=float),
        np.column_stack([df[c].to_numpy(dtype=float) for c in COVARIATES]),
        status=df["status"].to_numpy(dtype=int),
        strata=strata,
        ties="efron",
    ).fit()
    beta_ref = np.asarray(centralized.params, dtype=float)

    # Cleartext log-likelihood at the encrypted fit, for a like-for-like
    # comparison with the centralized model's own maximum.
    ll_enc = -sum(local_cox_nll(sites[n], beta_hat) for n in names)
    ll_ref = -sum(local_cox_nll(sites[n], beta_ref) for n in names)

    return CoxResult(
        backend=backend,
        coefficients=dict(zip(COVARIATES, beta_hat.tolist(), strict=True)),
        centralized=dict(zip(COVARIATES, beta_ref.tolist(), strict=True)),
        max_abs_difference=float(np.max(np.abs(beta_hat - beta_ref))),
        loglik_encrypted=float(ll_enc),
        loglik_centralized=float(ll_ref),
        objective_at_fit=float(fit.fun),
        n_objective_calls=calls["n"],
        converged=bool(fit.success),
        site_sizes={n: int(len(sites[n]["time"])) for n in names},
        site_events={n: int(sites[n]["status"].sum()) for n in names},
        master=master,
        context=ctx,
    )


if __name__ == "__main__":  # pragma: no cover
    for backend in ("ckks", "threshold"):
        r = run(backend)
        print(f"=== {backend} ===")
        print(f"sites   : {r.site_sizes} (events {r.site_events})")
        print(f"{'coefficient':<12} {'encrypted':>12} {'centralized':>13} {'diff':>10}")
        for row in r.table():
            print(
                f"{row['coefficient']:<12} {row['distributed_encrypted']:>12.6f} "
                f"{row['aggregated_cleartext']:>13.6f} {row['abs_diff']:>10.2e}"
            )
        print(f"max |diff|     : {r.max_abs_difference:.3e}")
        print(
            f"logLik encrypted / centralized: "
            f"{r.loglik_encrypted:.6f} / {r.loglik_centralized:.6f}"
        )
        print(f"objective calls: {r.n_objective_calls}, converged={r.converged}\n")
