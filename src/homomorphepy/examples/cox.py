"""Distributed Cox regression through an encrypted channel.

Both trust models live here because the whole point is that the
optimizer-facing code is *identical*: only the master class changes,
from a single-decrypter CKKS master to an n-of-n threshold one.
:func:`run` takes ``backend`` and the rest of the protocol is
untouched.

The setting is the Rosenwald DLBCL cohort, with each molecular
subgroup -- GCB, ABC and Type III -- treated as a site. The three sites
differ in size; the protocol does not require equal sizes.

Each site computes its own Cox partial log-likelihood at the current
coefficient vector and never shares patient rows. The site-level
function is ``PHReg(ties='efron').loglike(beta)``, checked against an
independent implementation of the stratified partial likelihood in
``tests/test_cox_loglik.py`` before this module was written.

``ties='efron'`` is not optional: ``PHReg`` defaults to Breslow, and on
this cohort the two conventions differ by up to 3.56 in pooled
log-likelihood -- a silent wrong answer rather than an error.
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
    make_ckks_master,
    make_threshold_master,
    make_worker,
)
from homomorphepy.context import Context, fhe_context
from homomorphepy.fixtures import load_dlbcl, site_order

__all__ = ["COVARIATES", "FTOL", "CoxResult", "local_cox_nll", "run"]

COVARIATES = ["GCB_sig", "LN_sig", "Prolif_sig", "BMP6", "MHC2_sig"]

# CKKS represents the summed stratified Cox nLL on this cohort at the
# default scaling parameters; scaling_mod_size is raised from 50 to 59
# for extra precision, and first_mod_size = 60, the library default, is
# set explicitly. The cox page prints the nLL at beta = 0 and at the MLE.
CKKS_PARAMS = dict(
    multiplicative_depth=1, scaling_mod_size=59, first_mod_size=60, batch_size=8
)

# No finite-difference step is set. Sweeping 1e-3 / 1e-5 / scipy's
# default changed neither the evaluation count nor the result by a
# single digit -- L-BFGS-B's path is insensitive to it on this problem.
# Carrying a constant that demonstrably does nothing would just be one
# more thing to explain. (Separately measured: the encrypted objective
# is accurate to ~1e-13, so the step could be shrunk freely if it ever
# did matter -- the CKKS noise floor is not the binding constraint
# here. Under differential privacy it is; see :mod:`.dp`.)

# The convergence criterion, and why it is L-BFGS-B rather than BFGS.
#
# `BFGS` tests the GRADIENT NORM (gtol): stop when the gradient is
# small. That threshold is not scale-free -- what counts as a small
# gradient depends on the objective's magnitude, so a value tuned on
# this cohort (nLL ~ 5e2) would silently need re-deriving for another
# dataset, and one chosen too small sits BELOW the attainable floor set
# by finite-difference truncation error and reports success=False
# forever. Measured here: gtol = 1e-6 failed that way on the CLEARTEXT
# objective too, so the failure was never about encryption.
#
# `L-BFGS-B`'s ftol is a RELATIVE FUNCTION-VALUE test,
#     (f_k - f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) <= ftol,
# which carries no dependence on the objective's scale. That is the
# reason for the choice: it transfers to another cohort unchanged.
#
# Swept against the centralized fit on this cohort:
#
#     ftol            max|b - b_centralized|   in SE units   evals
#     1e-7                    1.32e-04            1.1e-03      54
#     2.2e-9 (scipy default)  1.29e-05            1.1e-04      60
#     1e-9                    1.77e-07            1.7e-06      66
#     1e-12                   1.90e-07            1.3e-06      72
#
# 1e-9 reaches the accuracy floor: 1e-12 buys nothing, which says the
# limit is the finite-difference gradient rather than the tolerance.
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

    Plain statistical code that never touches an encrypted value. Returns
    NaN when the local computation fails, which the master propagates
    as a non-evaluable parameter.
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
        """Side-by-side comparison of the two fits, row per coefficient."""
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

    ``backend='ckks'`` uses a single-decrypter master; ``'threshold'``
    uses n-of-n threshold keys where no party can decrypt alone.
    Everything after the master is constructed is identical between
    the two.
    """
    sites = _site_frames()
    names = list(sites)

    ctx = fhe_context("CKKS", **CKKS_PARAMS)

    # The same worker construction serves both protocols: what makes a
    # site a threshold party is having run a key-generation round, not
    # being of a different type. Only the wiring differs — a
    # CKKSMaster is built first and takes workers, while the joint key
    # cannot exist before the sites do.
    workers = [make_worker(n, sites[n], local_cox_nll) for n in names]
    if backend == "ckks":
        master: Master = make_ckks_master("Master", ctx, ctx.KeyGen())
        master.set_workers(workers)
    elif backend == "threshold":
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
