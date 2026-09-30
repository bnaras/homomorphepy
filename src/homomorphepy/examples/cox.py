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

__all__ = ["COVARIATES", "FTOL", "CoxResult", "fd_hessian", "local_cox_nll", "run"]

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


HESSIAN_STEP = 1e-3


def fd_hessian(f, x: np.ndarray, h: float = HESSIAN_STEP) -> np.ndarray:
    """Central-difference Hessian of ``f`` at ``x``.

    For a fit through the encrypted channel, every evaluation here is a
    protocol round, as it is for the optimizer.
    """
    x = np.asarray(x, dtype=float)
    p = x.size
    e = np.eye(p) * h
    f0 = f(x)
    H = np.empty((p, p))
    for i in range(p):
        H[i, i] = (f(x + e[i]) - 2 * f0 + f(x - e[i])) / h**2
        for j in range(i):
            H[i, j] = H[j, i] = (
                f(x + e[i] + e[j])
                - f(x + e[i] - e[j])
                - f(x - e[i] + e[j])
                + f(x - e[i] - e[j])
            ) / (4 * h**2)
    return H


@dataclass
class CoxResult:
    backend: str
    coefficients: dict[str, float]
    std_errors: dict[str, float]
    cleartext: dict[str, float]
    centralized: dict[str, float]
    max_abs_difference: float
    max_abs_vs_centralized: float
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
        """The encrypted fit against the same objective fit in the clear.

        Row per coefficient. Both fits use the same optimizer, start and
        tolerance; only the aggregation differs.
        """
        return [
            {
                "coefficient": name,
                "encrypted": self.coefficients[name],
                "cleartext": self.cleartext[name],
                "abs_diff": abs(self.coefficients[name] - self.cleartext[name]),
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

    x0 = np.zeros(len(COVARIATES)) if start is None else np.asarray(start)
    fit = minimize(encrypted_nll, x0=x0, method="L-BFGS-B", options={"ftol": FTOL})
    beta_hat = np.asarray(fit.x, dtype=float)
    n_calls = calls["n"]

    # Standard errors from the Hessian of the encrypted objective at the
    # fit, each evaluation another protocol round.
    se = np.sqrt(np.diag(np.linalg.inv(fd_hessian(encrypted_nll, beta_hat))))

    # The same objective, the same optimizer, start and tolerance, with
    # the encrypted aggregation replaced by an ordinary sum. The
    # difference from the encrypted fit is what encryption changed.
    def cleartext_nll(beta):
        value = sum(local_cox_nll(sites[n], beta) for n in names)
        return PENALTY if math.isnan(value) else float(value)

    plain = minimize(cleartext_nll, x0=x0, method="L-BFGS-B", options={"ftol": FTOL})
    beta_plain = np.asarray(plain.x, dtype=float)

    # Centralized reference: one stratified fit on the pooled cohort by
    # a different algorithm (Newton, in statsmodels).
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
        std_errors=dict(zip(COVARIATES, se.tolist(), strict=True)),
        cleartext=dict(zip(COVARIATES, beta_plain.tolist(), strict=True)),
        centralized=dict(zip(COVARIATES, beta_ref.tolist(), strict=True)),
        max_abs_difference=float(np.max(np.abs(beta_hat - beta_plain))),
        max_abs_vs_centralized=float(np.max(np.abs(beta_hat - beta_ref))),
        loglik_encrypted=float(ll_enc),
        loglik_centralized=float(ll_ref),
        objective_at_fit=float(fit.fun),
        n_objective_calls=n_calls,
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
        print(f"{'coefficient':<12} {'encrypted':>12} {'cleartext':>13} {'diff':>10}")
        for row in r.table():
            print(
                f"{row['coefficient']:<12} {row['encrypted']:>12.6f} "
                f"{row['cleartext']:>13.6f} {row['abs_diff']:>10.2e}"
            )
        print(f"max |enc - cleartext|  : {r.max_abs_difference:.3e}")
        print(f"max |enc - centralized|: {r.max_abs_vs_centralized:.3e}")
        print(
            f"logLik encrypted / centralized: "
            f"{r.loglik_encrypted:.6f} / {r.loglik_centralized:.6f}"
        )
        print(f"objective calls: {r.n_objective_calls}, converged={r.converged}\n")
