"""Poisson MLE driven through an encrypted channel.

Three sites hold counts of the same adverse event. Nobody will share
their counts, but all are willing to compute the joint
maximum-likelihood estimate of the rate.

The point of the example is that **the optimizer is not modified**. It
is handed an objective that happens to route through encryption --
each evaluation encrypts three per-site log-likelihoods, sums them
homomorphically, and decrypts the total -- and it converges to the same
estimate as the pooled cleartext fit.

On the finite-difference step
-----------------------------

A finite-difference gradient divides a difference of objective values
by the step ``h``, so noise in the objective is amplified by ``1/h``.
SciPy's default step is ``sqrt(eps) ~ 1.5e-8``, small enough that it
is reasonable to expect the CKKS noise floor to swamp the gradient and
leave BFGS differentiating noise.

**Measured, that does not happen here.** At depth 1 and
``scaling_mod_size = 50``, on an objective of magnitude ~100:

===========================  =========
decrypted absolute error     ~1.1e-13
spread over repeated evals   ~2.0e-13
implied gradient noise
  at SciPy's default step    ~1.7e-06
  at a 1e-3 step             ~2.5e-11
true gradient at lambda = 9  -0.78
===========================  =========

The gradient signal is five to six orders of magnitude above the noise
even at the default step, and both settings converge: the default
lands 9.2e-07 from the pooled estimate, the widened step 2.6e-07. CKKS
at these parameters is far more precise than one might assume --
1e-13 absolute on a value of 100 is essentially float64 precision.

:data:`FINITE_DIFF_STEP` widens the step anyway, since it is
marginally more accurate, but it is not load-bearing. A deeper computation
with a larger scaling factor could change the balance, so
``tests/test_examples.py`` records the measurement and will flag it if
the two settings ever stop agreeing. The step *does* become
load-bearing under differential privacy, where the injected noise is
many orders larger than CKKS's.

One further wrinkle: a parameter outside the support makes the
objective non-evaluable, and SciPy's line search cannot act on NaN.
:func:`make_objective` converts that into a large finite penalty,
which the line search *can* step back from.
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

__all__ = [
    "FINITE_DIFF_STEP",
    "PENALTY",
    "N",
    "TRUE_LAMBDA",
    "SITE_SPLIT",
    "MLEResult",
    "simulate",
    "local_nll",
    "make_objective",
    "run",
]

N = 40  # total counts across all sites
TRUE_LAMBDA = 10.0  # the rate the data is drawn at
SITE_SPLIT = (20, 7, 13)  # how those counts are distributed

# Wider than SciPy's default sqrt(eps) ~ 1.5e-8. Marginally more
# accurate here, but measured NOT required -- see the module docstring.
FINITE_DIFF_STEP = 1e-3


def simulate(seed: int = 17822) -> list[np.ndarray]:
    """Poisson counts, partitioned across three sites.

    One pooled draw split into three unequal pieces, mirroring three
    hospitals of different sizes counting the same adverse event.
    """
    rng = np.random.default_rng(seed)
    y = rng.poisson(TRUE_LAMBDA, size=N)
    bounds = np.cumsum((0,) + SITE_SPLIT)
    return [y[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]


# Returned for a parameter no site can evaluate. Finite, so the line
# search backs off rather than propagating NaN through the Wolfe tests.
PENALTY = 1e12


def local_nll(data: Sequence[int], lam: float) -> float:
    """Negative Poisson log-likelihood of one site's counts.

    Plain numeric code, exactly as it would be written without any
    encryption -- the site never sees an encrypted value.
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


def run(start: float = 5.0, seed: int = 17822) -> MLEResult:
    """Fit the Poisson rate through the encrypted channel."""
    sites_data = simulate(seed)

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
    # unreliable as a variance estimate.
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
