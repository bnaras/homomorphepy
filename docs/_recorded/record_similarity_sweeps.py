"""Record the adapter sweeps for the documentation.

Each sweep point is a federated retrieval over three sites, averaged
over ``n_rep`` replicates of forty queries, and the mu sweep does that
at five values of mu plus three reference lines. Running it at render
time would add minutes to every documentation build, so this script
runs it once and writes ``similarity_sweeps.json``;
``docs/similarity.qmd`` reads that file and plots it.

The sweeps run **in the clear**. That is a deliberate choice, not a
shortcut: ``similarity.run()`` measures the encrypted and cleartext
scores against each other on the same protocol and they agree to
~1e-13 with identical ranking, so pushing three replicates x five mu x
forty queries through threshold decryption would cost hours and change
no digit of a recall@5. The encrypted claim is established once, by
the page's main run; the sweeps answer a separate question -- which
adapter, and how much regularization -- that encryption does not bear
on.

Run from the repository root::

    OMP_NUM_THREADS=2 uv run python docs/_recorded/record_similarity_sweeps.py

Nothing here is hidden: every number and every curve in the rendered
page comes from this script, and re-running it regenerates them.
"""

from __future__ import annotations

import json
import math
import time
from datetime import date
from pathlib import Path

from homomorphepy.examples import similarity

OUT = Path(__file__).with_name("similarity_sweeps.json")

# Ample calibration: the anchor cohort is larger than the embedding
# dimension, so the adapter is determined and mu only trades fidelity
# against isometry.
N_ANCHOR_AMPLE = 100

# Scarce calibration: fewer anchors than p = 32, so the least-squares
# end of the family is underdetermined and an intermediate mu can beat
# both endpoints. This is the panel where the sweet spot appears.
N_ANCHOR_SCARCE = 24

BETA = 0.6


def _mu_label(mu: float) -> str:
    """Axis label for a mu grid point.

    Stored as a string rather than a float because ``inf`` is not
    representable in strict JSON, and because the plot's x-axis is
    categorical anyway -- the grid is not evenly spaced in mu.
    """
    return "inf" if math.isinf(mu) else f"{mu:g}"


def _pack(sweep: dict) -> dict:
    return {
        "beta": sweep["beta"],
        "n_anchor": sweep["n_anchor"],
        "mu_labels": [_mu_label(r["mu"]) for r in sweep["table"]],
        "d1": [r["d1"] for r in sweep["table"]],
        "d2": [r["d2"] for r in sweep["table"]],
        "aniso": [r["aniso"] for r in sweep["table"]],
        "ideal": sweep["ideal"],
        "unaligned": sweep["unaligned"],
        "gram": sweep["gram"],
    }


def main() -> None:
    started = time.monotonic()

    print(f"mu sweep, ample anchor (n = {N_ANCHOR_AMPLE}) ...")
    mu_ample = similarity.mu_sweep(beta=BETA, n_anchor=N_ANCHOR_AMPLE)

    print(f"mu sweep, scarce anchor (n = {N_ANCHOR_SCARCE} < p) ...")
    mu_scarce = similarity.mu_sweep(beta=BETA, n_anchor=N_ANCHOR_SCARCE)

    print("beta sweep ...")
    beta_rows = similarity.beta_sweep(n_anchor=N_ANCHOR_AMPLE)

    minutes = (time.monotonic() - started) / 60.0
    OUT.write_text(
        json.dumps(
            {
                "provenance": (
                    "docs/_recorded/record_similarity_sweeps.py, "
                    f"{date.today().isoformat()}"
                ),
                "wall_clock_minutes": round(minutes, 1),
                "config": {
                    k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in similarity.SWEEP.items()
                    if k != "mu_grid"
                },
                "mu_grid": [_mu_label(m) for m in similarity.SWEEP["mu_grid"]],
                "mu_ample": _pack(mu_ample),
                "mu_scarce": _pack(mu_scarce),
                "beta_sweep": beta_rows,
            },
            indent=2,
        )
        + "\n"
    )

    for row in mu_ample["table"]:
        print(
            f"  mu={_mu_label(row['mu']):>4}  d1={row['d1']:.3f} "
            f"d2={row['d2']:.3f}  aniso={row['aniso']:.2e}"
        )
    print(
        f"  ideal={mu_ample['ideal']:.3f}  "
        f"unaligned={mu_ample['unaligned']:.3f}  gram={mu_ample['gram']:.3f}"
    )
    print(f"wrote {OUT} after {minutes:.1f} min")


if __name__ == "__main__":
    main()
