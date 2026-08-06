"""Distributed query count under threshold BFV.

Ports homomorpheR's ``query-count-threshold.Rmd``, which itself
re-derives the old Paillier non-cooperating-parties demonstration
using threshold FHE.

Several sites each hold a private table of patient records. A
coordinator wants the answer to one aggregate query -- how many
patients across all sites satisfy some condition -- without any site
revealing its records and without the coordinator learning any
individual site's count.

The cohort is **simulated here**, not imported from R. The data is
drawn from a data-generating process, so there is no external truth to
match: what matters is that the encrypted protocol reproduces the
pooled cleartext answer *on whatever data this run produced*. That is
internal consistency, and it is the claim the example makes.

Cross-language agreement on values is reserved for the examples built
on the real DLBCL cohort (:mod:`.cox`, :mod:`.cox_lasso`, :mod:`.dp`),
where the data is given rather than drawn and both languages must be
reading the same measurements.

BFV keeps the count exact, so the comparison is an equality of
integers rather than an agreement within tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from homomorphepy.actors import ThresholdMaster, ThresholdSite, make_threshold_master
from homomorphepy.context import Context, fhe_context

__all__ = ["QueryCountResult", "QUERY", "simulate", "count_fn", "run"]

PLAINTEXT_MODULUS = 65537
SITE_SIZES = (60, 15, 25)
QUERY = 'age < 50 and sex == "F" and bm < 0.2'


def simulate(seed: int = 130):
    """Three site tables of patient records.

    The same data-generating process as the R vignette -- sex, age in
    [40, 70], and a standard-normal biomarker -- but drawn with numpy.
    R's ``sample`` and ``rnorm`` are different algorithms, so this is
    not R's draw at any seed, and it does not need to be.
    """
    rng = np.random.default_rng(seed)
    sites, start = [], 1
    for n in SITE_SIZES:
        sites.append(
            {
                "id": [f"P{i:4d}" for i in range(start, start + n)],
                "sex": rng.choice(["F", "M"], size=n).tolist(),
                "age": rng.integers(40, 71, size=n).tolist(),
                "bm": rng.normal(size=n).tolist(),
            }
        )
        start += n
    return sites


def count_fn(rows: dict[str, list[Any]], _theta: Any = None) -> int:
    """Evaluate the query on one site's rows.

    R passes a quoted expression evaluated with ``eval(query, data)``;
    spelling the predicate out keeps the two sides comparable without
    importing an expression-evaluation mechanism with no R counterpart.
    """
    return sum(
        1
        for i in range(len(rows["age"]))
        if rows["age"][i] < 50 and rows["sex"][i] == "F" and rows["bm"][i] < 0.2
    )


@dataclass
class QueryCountResult:
    total_encrypted: int
    total_cleartext: int
    per_site_cleartext: list[int]
    site_sizes: list[int]
    query: str
    context: Context = field(repr=False)
    master: ThresholdMaster = field(repr=False)

    @property
    def exact(self) -> bool:
        """BFV is exact arithmetic; this must always hold."""
        return self.total_encrypted == self.total_cleartext


def run(seed: int = 130) -> QueryCountResult:
    """Run the threshold-BFV query count over a simulated cohort."""
    sites_data = simulate(seed)

    ctx = fhe_context(
        "BFV", plaintext_modulus=PLAINTEXT_MODULUS, multiplicative_depth=1
    )
    sites = [
        ThresholdSite(f"site{i + 1}", rows, count_fn)
        for i, rows in enumerate(sites_data)
    ]
    # Chained key generation: every site ends up with its own share and
    # the master with only the joint public key.
    master = make_threshold_master("Coordinator", ctx, sites)

    total = master.aggregate(None)
    per_site = [count_fn(rows) for rows in sites_data]

    return QueryCountResult(
        total_encrypted=int(total),
        total_cleartext=sum(per_site),
        per_site_cleartext=per_site,
        site_sizes=[len(r["age"]) for r in sites_data],
        query=QUERY,
        context=ctx,
        master=master,
    )


if __name__ == "__main__":  # pragma: no cover
    r = run()
    print(f"query            : {r.query}")
    print(f"site sizes       : {r.site_sizes}")
    print(f"per-site counts  : {r.per_site_cleartext}  (no party sees these)")
    print(f"encrypted total  : {r.total_encrypted}")
    print(f"pooled cleartext : {r.total_cleartext}")
    print(f"exact            : {r.exact}")
    print(f"sites holding a share: {len(r.master.sites)}; "
          f"master holds none: {not hasattr(r.master, 'secret_share')}")
