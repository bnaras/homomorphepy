"""Poisson MLE driven through an encrypted channel.

Ported from homomorpheR's ``mle.Rmd``. Three sites hold counts of the
same adverse event. Nobody will share their counts, but all are willing
to compute the joint maximum-likelihood estimate of the rate.

The point of the example is that **the optimizer is not modified**. It
is handed an objective that happens to route through encryption --
each evaluation encrypts three per-site log-likelihoods, sums them
homomorphically, and decrypts the total -- and it converges to the same
estimate as the pooled cleartext fit.

On the finite-difference step
-----------------------------

A review predicted this would be a problem: R's ``optim`` builds its
numeric gradient with ``ndeps = 1e-3``, whereas SciPy's default step is
``sqrt(eps) ~ 1.5e-8``, and the argument was that the smaller step
would fall below the CKKS noise floor and leave BFGS differentiating
noise.

**Measured, that does not happen here.** At depth 1 and
``scaling_mod_size = 50``, on an objective of magnitude ~100:

===========================  =========
decrypted absolute error     ~1.1e-13
spread over repeated evals   ~2.0e-13
implied gradient noise
  at SciPy's default step    ~1.7e-06
  at R's 1e-3 step           ~2.5e-11
true gradient at lambda = 9  -0.78
===========================  =========

The gradient signal is five to six orders of magnitude above the noise
even at the default step, and both settings converge: the default
lands 9.2e-07 from the pooled estimate, the widened step 2.6e-07. CKKS
at these parameters is far more precise than the prediction assumed --
1e-13 absolute on a value of 100 is essentially float64 precision.

:data:`FINITE_DIFF_STEP` still matches R's ``ndeps``, for
cross-language comparability rather than out of necessity, and because
it is marginally more accurate. It is not load-bearing. A deeper
circuit with a larger scaling factor could change the balance, so
``tests/test_examples.py`` records the measurement and will flag it if
the two settings ever stop agreeing.

The other difference is the non-evaluable case. R's optimizer backs off
when the objective returns ``NA``; SciPy's line search does not, and a
``None`` objective raises ``TypeError``. :func:`make_objective`
therefore converts a non-evaluable parameter into a large finite
penalty, which the line search *can* act on.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm

from homomorphepy.actors import CKKSMaster, make_ckks_master, make_worker
from homomorphepy.context import Context, fhe_context
from homomorphepy.fixtures import load_json

__all__ = [
    "FINITE_DIFF_STEP",
    "PENALTY",
    "MLEResult",
    "local_nll",
    "make_objective",
    "run",
]

# Matches R's optim(control = list(ndeps = 1e-3)), for cross-language
# comparability. Measured, NOT required -- see the module docstring.
FINITE_DIFF_STEP = 1e-3

# Returned for a parameter no site can evaluate. Finite, so the line
# search backs off rather than propagating NaN through the Wolfe tests.
PENALTY = 1e12


def local_nll(data: Sequence[int], lam: float) -> float:
    """Negative Poisson log-likelihood of one site's counts.

    Plain numeric code, exactly as it would be written without any
    encryption -- the site never sees a ciphertext.
    """
    lam = float(np.ravel(lam)[0])
    if lam <= 0:
        return math.nan  # outside the parameter space
    y = np.asarray(data, dtype=float)
    return float(-np.sum(y * math.log(lam) - lam - gammaln(y + 1.0)))


def make_objective(master: CKKSMaster):
    """Wrap the encrypted aggregate as a SciPy-compatible objective."""

    def objective(theta):
        value = master.aggregate(float(np.ravel(theta)[0]))
        # aggregate() returns NaN when a site cannot evaluate; SciPy's
        # line search cannot act on NaN, so convert to a large finite
        # value it can reject and step back from.
        return PENALTY if (value is None or math.isnan(value)) else float(value)

    return objective


@dataclass
class MLEResult:
    lambda_encrypted: float
    lambda_pooled: float
    std_error: float
    n_total: int
    site_sizes: list[int]
    n_objective_calls: int
    converged: bool
    context: Context = field(repr=False)

    @property
    def abs_difference(self) -> float:
        """|encrypted estimate - pooled cleartext estimate|."""
        return abs(self.lambda_encrypted - self.lambda_pooled)

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        z = float(norm.ppf(0.5 + level / 2))
        return (
            self.lambda_encrypted - z * self.std_error,
            self.lambda_encrypted + z * self.std_error,
        )


def run(start: float = 5.0) -> MLEResult:
    """Fit the Poisson rate through the encrypted channel."""
    fixture = load_json("mle_poisson")
    sites_data = [s["y"] for s in fixture["sites"]]

    ctx = fhe_context("CKKS", multiplicative_depth=1, scaling_mod_size=50, batch_size=8)
    keys = ctx.KeyGen()

    workers = [
        make_worker(f"Site {i + 1}", data, local_nll)
        for i, data in enumerate(sites_data)
    ]
    master = make_ckks_master("Master", ctx, keys).set_workers(workers)

    objective = make_objective(master)
    calls = {"n": 0}

    def counted(theta):
        calls["n"] += 1
        return objective(theta)

    fit = minimize(
        counted,
        x0=[start],
        method="BFGS",
        options={"finite_diff_rel_step": FINITE_DIFF_STEP},
    )
    lam_hat = float(fit.x[0])

    # Pooled cleartext reference. For Poisson the MLE is the sample
    # mean in closed form, so this is exact rather than another
    # numerical fit -- a stronger comparison than optimizer-vs-optimizer.
    pooled = np.concatenate([np.asarray(d, dtype=float) for d in sites_data])
    lam_pooled = float(pooled.mean())

    # Standard error from the analytic Fisher information, n/lambda.
    # NOT from fit.hess_inv: the BFGS inverse-Hessian approximation is
    # unreliable as a variance estimate (and R's SE comes from a
    # numeric Hessian at ndeps=1e-3, not from the quasi-Newton state).
    n = int(pooled.size)
    se = float(math.sqrt(lam_hat / n))

    return MLEResult(
        lambda_encrypted=lam_hat,
        lambda_pooled=lam_pooled,
        std_error=se,
        n_total=n,
        site_sizes=[len(d) for d in sites_data],
        n_objective_calls=calls["n"],
        converged=bool(fit.success),
        context=ctx,
    )


if __name__ == "__main__":  # pragma: no cover
    r = run()
    lo, hi = r.ci()
    print(f"sites             : {r.site_sizes} (n = {r.n_total})")
    print(f"encrypted MLE     : {r.lambda_encrypted:.6f}")
    print(f"pooled cleartext  : {r.lambda_pooled:.6f}")
    print(f"difference        : {r.abs_difference:.2e}")
    print(f"std error         : {r.std_error:.6f}   95% CI ({lo:.4f}, {hi:.4f})")
    print(
        f"objective calls   : {r.n_objective_calls}  (each one a full "
        f"encrypt/sum/decrypt round)"
    )
