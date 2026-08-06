"""Record the DP noise sweep for the documentation.

Fitting the threshold-Cox model once per (sigma, optimizer) pair takes
a few minutes, and the gradient-free arm needs thousands of objective
evaluations -- each one a full encrypt / sum / threshold-decrypt round.
That is too slow for every documentation render, so this script runs
the sweep once and writes ``dp_sweep.json``; ``docs/dp.qmd`` reads that
file rather than carrying hand-typed numbers.

Run from the repository root::

    OMP_NUM_THREADS=2 uv run python docs/_recorded/record_dp_sweep.py

Nothing here is hidden: the table in the rendered page comes from this
script, and re-running it regenerates it.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

from homomorphepy.examples import dp

OUT = Path(__file__).with_name("dp_sweep.json")

SIGMAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
METHODS = ("BFGS", "Nelder-Mead")

# A fit that returns its starting point has not fitted anything, and
# the optimizer still reports success -- so collapse is detected from
# the iterate, not the status flag.
#
# The test is RELATIVE to the centralized coefficient scale. An
# absolute epsilon does not work: a collapsed fit here leaves x0 = 0
# perturbed to ~1e-5 by the noisy line search, which is far from zero
# in absolute terms but is 0.003% of the smallest real coefficient.
# Anything under 1% of the largest centralized coefficient has plainly
# not moved off the origin.
COLLAPSE_FRACTION = 0.01


def main() -> None:
    started = time.monotonic()
    rows = []
    for sigma in SIGMAS:
        for method in METHODS:
            fit = dp.fit_at_sigma(sigma, method=method, seed=1)
            beta = list(fit.coefficients.values())
            reference_scale = max(abs(v) for v in fit.centralized.values())
            rows.append(
                {
                    "sigma": sigma,
                    "method": method,
                    "max_abs_diff": fit.max_abs_diff,
                    "sign_agreement": fit.sign_agreement,
                    "n_queries": fit.n_queries,
                    "epsilon": fit.epsilon,
                    "coefficients": fit.coefficients,
                    "max_abs_beta": max(abs(b) for b in beta),
                    "reference_scale": reference_scale,
                    "collapsed": (
                        max(abs(b) for b in beta)
                        < COLLAPSE_FRACTION * reference_scale
                    ),
                }
            )
            print(
                f"sigma={sigma:<7g} {method:<12} "
                f"diff={fit.max_abs_diff:.3f} signs={fit.sign_agreement}/5 "
                f"queries={fit.n_queries:<5} "
                f"max|beta|={max(abs(b) for b in beta):.2e}"
            )

    minutes = (time.monotonic() - started) / 60.0
    OUT.write_text(
        json.dumps(
            {
                "provenance": (
                    f"docs/_recorded/record_dp_sweep.py, {date.today().isoformat()}"
                ),
                "wall_clock_minutes": round(minutes, 1),
                "collapse_fraction": COLLAPSE_FRACTION,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT} after {minutes:.1f} min")


if __name__ == "__main__":
    main()
