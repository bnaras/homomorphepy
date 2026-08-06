"""Record the full Cox-lasso run for the documentation.

The consensus ADMM at K = 100 needs three per-site conic solves per
iteration and takes about half an hour, which is too long to run on
every documentation render. This script runs it once and writes the
results to ``cox_lasso_run.json``; ``docs/cox-lasso.qmd`` reads that
file and never hard-codes a value.

Run from the repository root::

    OMP_NUM_THREADS=2 uv run python docs/_recorded/record_cox_lasso.py

Nothing here is hidden: the numbers in the rendered page come from
this script, and re-running it regenerates them.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np

from homomorphepy.examples import cox_lasso

OUT = Path(__file__).with_name("cox_lasso_run.json")


def main() -> None:
    started = time.monotonic()
    result = cox_lasso.run(recompute_admm=True)
    minutes = (time.monotonic() - started) / 60.0

    payload = {
        "provenance": (
            f"docs/_recorded/record_cox_lasso.py, {date.today().isoformat()}"
        ),
        "wall_clock_minutes": round(minutes, 1),
        "n_probes_total": 6416,
        "n_probes_screened": int(result.top_idx.size),
        "n_iter": int(result.n_iter),
        "n_nonzero": int(result.n_nonzero),
        "admm_vs_centralized": float(result.admm_vs_centralized),
        "pool_agree_mu": float(result.pool_agree_mu),
        "pool_agree_sigma": float(result.pool_agree_sigma),
        "beta_admm": np.asarray(result.beta_admm, dtype=float).tolist(),
        "beta_centralized": np.asarray(
            result.beta_centralized, dtype=float
        ).tolist(),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT} after {minutes:.1f} min")


if __name__ == "__main__":
    main()
