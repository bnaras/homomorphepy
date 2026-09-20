"""Logistic prediction evaluated entirely under encryption.

The secure-inference example scored patients with a *linear* model. A
logistic model needs the sigmoid

.. math:: \\sigma(\\eta) = 1 / (1 + e^{-\\eta})

which is not a polynomial, and encrypted arithmetic offers only
addition and multiplication. A Chebyshev approximation closes that gap:
the linear predictor *and* the sigmoid are evaluated without ever
decrypting, so the hospital never sees the coefficients and the
researcher never sees the patients.

The cost is paid in precision budget. Every multiplication in the
polynomial spends one level, which is why the context below asks for
depth 8 where the linear example needed 2.

Where the numbers come from
---------------------------

The training cohort is a shipped fixture rather than a fresh draw, so
this example and its R counterpart fit the same 500 patients and can be
compared coefficient by coefficient. The scoring panel of 16 patients
is a fixed table for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from homomorphepy.ciphertext import Ct
from homomorphepy.codec import packed_codec
from homomorphepy.context import Context, backend, fhe_context
from homomorphepy.fixtures import load_json

__all__ = [
    "RegressionResult",
    "PANEL_AGE",
    "PANEL_BIOMARKER",
    "fit_logistic",
    "run",
]

# The 16 patients to be scored. A fixed panel, not a random draw: the
# example is about the pipeline, and fixed values keep the printed
# probabilities stable between runs and identical to the R vignette's.
PANEL_AGE = (45, 52, 60, 38, 70, 55, 48, 63, 41, 57, 66, 44, 72, 50, 59, 35)
PANEL_BIOMARKER = (
    -0.5,
    0.3,
    1.2,
    -1.0,
    0.8,
    0.1,
    -0.3,
    1.5,
    -0.8,
    0.6,
    0.9,
    -0.4,
    1.1,
    0.0,
    0.7,
    -1.2,
)

# The sigmoid is approximated on this interval. It has to cover the
# range the linear predictor can reach on the panel; outside it the
# Chebyshev fit diverges quickly and silently.
CHEBYSHEV_INTERVAL = (-4.0, 4.0)
CHEBYSHEV_DEGREE = 16


@dataclass
class RegressionResult:
    beta: list[float]
    beta_reference: list[float]
    beta_max_diff: float
    probs_encrypted: list[float]
    probs_cleartext: list[float]
    max_error: float
    n_train: int
    n_scored: int
    context: Context = field(repr=False)


def fit_logistic(
    age: np.ndarray, biomarker: np.ndarray, outcome: np.ndarray
) -> np.ndarray:
    """Maximum-likelihood logistic fit, in the clear.

    Newton-Raphson directly rather than through a modelling library:
    the fit is three columns on 500 rows, and doing it here keeps the
    example free of a dependency it would use once.
    """
    X = np.column_stack([np.ones_like(age), age, biomarker])
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1.0 - mu)
        # Ridge term guards the solve, not the estimate: it is far below
        # the curvature of a well-separated three-parameter fit.
        hessian = X.T @ (W[:, None] * X) + 1e-10 * np.eye(X.shape[1])
        step = np.linalg.solve(hessian, X.T @ (outcome - mu))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def run() -> RegressionResult:
    """Fit in the clear, then score the panel without decrypting."""
    fx = load_json("encrypted_regression")
    age = np.asarray(fx["age"], dtype=float)
    biomarker = np.asarray(fx["biomarker"], dtype=float)
    outcome = np.asarray(fx["outcome"], dtype=float)

    beta = fit_logistic(age, biomarker, outcome)
    beta_reference = np.asarray(fx["expected"]["beta_fit"], dtype=float)

    # -- hospital: context, keys, encryption -------------------------
    ofhe = backend()
    ctx = fhe_context(
        "CKKS",
        multiplicative_depth=8,
        scaling_mod_size=50,
        batch_size=16,
        features=[ofhe.PKESchemeFeature.ADVANCEDSHE],
    )
    keys = ctx.KeyGen()
    ctx.EvalMultKeyGen(keys.secretKey)
    codec = packed_codec(ctx)

    panel_age = np.asarray(PANEL_AGE, dtype=float)
    panel_bm = np.asarray(PANEL_BIOMARKER, dtype=float)

    ct_age = Ct(ctx.Encrypt(keys.publicKey, codec.encode(panel_age.tolist())), ctx.cc)
    ct_bm = Ct(ctx.Encrypt(keys.publicKey, codec.encode(panel_bm.tolist())), ctx.cc)

    # -- researcher: linear predictor, then the sigmoid --------------
    # Reads as the cleartext expression would; nothing is decrypted.
    ct_eta = ct_age * float(beta[1]) + ct_bm * float(beta[2]) + float(beta[0])

    a, b = CHEBYSHEV_INTERVAL
    ct_prob = Ct(
        ctx.cc.EvalLogistic(ct_eta.raw, a, b, CHEBYSHEV_DEGREE),
        ctx.cc,
    )

    # -- hospital: decrypt -------------------------------------------
    n = len(PANEL_AGE)
    probs = np.asarray(
        codec.decode(ctx.Decrypt(ct_prob.raw, keys.secretKey), n), dtype=float
    )

    eta_clear = beta[0] + beta[1] * panel_age + beta[2] * panel_bm
    probs_clear = 1.0 / (1.0 + np.exp(-eta_clear))

    return RegressionResult(
        beta=beta.tolist(),
        beta_reference=beta_reference.tolist(),
        beta_max_diff=float(np.max(np.abs(beta - beta_reference))),
        probs_encrypted=probs.tolist(),
        probs_cleartext=probs_clear.tolist(),
        max_error=float(np.max(np.abs(probs - probs_clear))),
        n_train=int(fx["n"]),
        n_scored=n,
        context=ctx,
    )


if __name__ == "__main__":  # pragma: no cover
    r = run()
    print(f"trained on {r.n_train} patients, scored {r.n_scored}")
    print("  coefficients      :", " ".join(f"{v:.4f}" for v in r.beta))
    print(f"  vs shipped fit    : {r.beta_max_diff:.2e}")
    print(f"  max error vs cleartext sigmoid : {r.max_error:.2e}")
