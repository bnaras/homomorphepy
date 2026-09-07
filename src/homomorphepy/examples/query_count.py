"""Distributed query count under threshold BFV.

Several sites each hold a private table of patient records. A
aggregator wants the answer to one aggregate query -- how many
patients across all sites satisfy some condition -- without any site
revealing its records and without the aggregator learning any
individual site's count.

The query travels as a string and is evaluated at each site with
:meth:`pandas.DataFrame.query`, so the predicate is a parameter of the
protocol rather than something baked into the site function. The
aggregator broadcasts it; each site answers with an encrypted count.

The cohort is simulated, so there is no external truth to match: the
claim is that the encrypted protocol reproduces the pooled cleartext
answer *on whatever data this run produced*. BFV keeps the count
exact, so that comparison is an equality of integers rather than an
agreement within tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from homomorphepy.actors import ThresholdMaster, ThresholdSite, make_threshold_master
from homomorphepy.context import Context, fhe_context

__all__ = ["QueryCountResult", "QUERY", "SITE_SIZES", "simulate", "count_fn", "run"]

PLAINTEXT_MODULUS = 65537
SITE_SIZES = (60, 15, 25)
QUERY = 'age < 50 and sex == "F" and bm < 0.2'


def simulate(seed: int = 130) -> list[pd.DataFrame]:
    """Three site tables of patient records: sex, age, and a biomarker."""
    rng = np.random.default_rng(seed)
    sites, start = [], 1
    for n in SITE_SIZES:
        sites.append(
            pd.DataFrame(
                {
                    "id": [f"P{i:04d}" for i in range(start, start + n)],
                    "sex": rng.choice(["F", "M"], size=n),
                    "age": rng.integers(40, 71, size=n),
                    "bm": rng.normal(size=n),
                }
            )
        )
        start += n
    return sites


def count_fn(data: pd.DataFrame, query: str) -> int:
    """Evaluate the broadcast query against one site's private rows."""
    return int(len(data.query(query)))


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


def run(seed: int = 130, query: str = QUERY) -> QueryCountResult:
    """Run the threshold-BFV query count over a simulated cohort."""
    site_data = simulate(seed)

    ctx = fhe_context(
        "BFV", plaintext_modulus=PLAINTEXT_MODULUS, multiplicative_depth=1
    )
    sites = [
        ThresholdSite(f"Site {i + 1}", rows, count_fn)
        for i, rows in enumerate(site_data)
    ]
    # Chained key generation: every site ends up with its own share and
    # the master with only the joint public key.
    master = make_threshold_master("Aggregator", ctx, sites)

    total = master.aggregate(query)
    per_site = [count_fn(rows, query) for rows in site_data]

    return QueryCountResult(
        total_encrypted=int(total),
        total_cleartext=sum(per_site),
        per_site_cleartext=per_site,
        site_sizes=[len(r) for r in site_data],
        query=query,
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
    print(
        f"sites holding a share: {len(r.master.sites)}; "
        f"master holds none: {not hasattr(r.master, 'secret_share')}"
    )
