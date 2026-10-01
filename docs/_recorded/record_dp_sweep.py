"""Record the DP noise sweep for the documentation.

Fitting the threshold-Cox model once per (sigma, optimizer) pair takes
a few minutes, and the gradient-free arm needs thousands of objective
evaluations -- each one a full encrypt / sum / threshold-decrypt round.
That is too slow for every documentation render, so this script runs
the sweep once and writes ``dp_sweep.json``; ``docs/dp.qmd`` reads that
file rather than carrying hand-typed numbers.

Run from the repository root::

    OMP_NUM_THREADS=2 uv run python docs/_recorded/record_dp_sweep.py

Nothing here is hidden: the tables in the rendered page come from this
script, and re-running it regenerates them.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

from homomorphepy.examples import dp

OUT = Path(__file__).with_name("dp_sweep.json")

SIGMAS = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
METHODS = ("BFGS", "Nelder-Mead")


def main() -> None:
    started = time.monotonic()
    rows = []
    for method in METHODS:
        for sigma in SIGMAS:
            fit = dp.fit_at_sigma(sigma, method=method, seed=1)
            rows.append(
                {
                    "sigma": sigma,
                    "method": method,
                    "n_queries": fit.n_queries,
                    "coefficients": fit.coefficients,
                    "centralized": fit.centralized,
                    "max_abs_diff": fit.max_abs_diff,
                }
            )
            print(
                f"{method:<12} sigma={sigma:<7g} "
                f"diff={fit.max_abs_diff:.3g} queries={fit.n_queries}"
            )

    minutes = (time.monotonic() - started) / 60.0
    OUT.write_text(
        json.dumps(
            {
                "provenance": (
                    f"docs/_recorded/record_dp_sweep.py, {date.today().isoformat()}"
                ),
                "wall_clock_minutes": round(minutes, 1),
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT} after {minutes:.1f} min")


if __name__ == "__main__":
    main()
