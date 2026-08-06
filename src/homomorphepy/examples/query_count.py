"""Query counting under threshold BFV: no single party can decrypt.

Ported from homomorpheR's ``query-count-threshold.Rmd``, which itself
re-derives the old Paillier non-cooperating-parties demonstration using
threshold FHE.

The computation is the same as :mod:`.aggregation` -- count the
patients matching a query, across sites, without revealing per-site
counts. What changes is the trust model. There, one coordinator held
the secret key and could have decrypted an individual site's
ciphertext. Here the key is generated n-of-n: each site holds a share,
encryption is under the joint public key, and recovering the total
requires a partial decryption from *every* site. No party, including
the master, can decrypt anything alone.

The Paillier version needed two extra non-cooperating parties and an
additive masking dance to get a comparable guarantee. Threshold FHE
obtains it directly, which is why the NCP machinery is frozen (D5).

BFV keeps the count exact, so the result is asserted with ``==``
against the value R computed on the same rows -- a genuine
cross-language equality, not an agreement within tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from homomorphepy.actors import ThresholdMaster, ThresholdSite, make_threshold_master
from homomorphepy.context import Context, fhe_context
from homomorphepy.fixtures import load_json

__all__ = ["QueryCountResult", "run", "QUERY"]

PLAINTEXT_MODULUS = 65537
QUERY = 'age < 50 and sex == "F" and bm < 0.2'


@dataclass
class QueryCountResult:
    total_encrypted: int
    total_cleartext: int
    per_site_cleartext: list[int]
    r_expected_total: int
    r_expected_per_site: list[int]
    query: str
    context: Context
    master: ThresholdMaster

    @property
    def matches_r(self) -> bool:
        """Exact agreement with the R implementation on the same rows."""
        return (
            self.total_encrypted == self.r_expected_total
            and self.per_site_cleartext == self.r_expected_per_site
        )


def count_fn(rows: dict[str, list[Any]], _theta: Any = None) -> int:
    """Evaluate the query on one site's rows.

    R passes a quoted expression evaluated with ``eval(query, data)``;
    the Python analogue would be a ``DataFrame.query`` string. Spelling
    the predicate out keeps the two sides comparable without importing
    an expression-evaluation mechanism that has no R counterpart.
    """
    return sum(
        1
        for i in range(len(rows["age"]))
        if rows["age"][i] < 50 and rows["sex"][i] == "F" and rows["bm"][i] < 0.2
    )


def run() -> QueryCountResult:
    """Run the threshold-BFV query count over the R-exported cohorts."""
    fixture = load_json("query_count")

    ctx = fhe_context(
        "BFV", plaintext_modulus=PLAINTEXT_MODULUS, multiplicative_depth=1
    )

    sites = [
        ThresholdSite(f"site{i + 1}", rows, count_fn)
        for i, rows in enumerate(fixture["sites"])
    ]
    # Chained key generation: each site ends up with its own share and
    # the master with only the joint public key.
    master = make_threshold_master("Coordinator", ctx, sites)

    total = master.aggregate(None)

    return QueryCountResult(
        total_encrypted=int(total),
        total_cleartext=sum(count_fn(s.data) for s in sites),
        per_site_cleartext=[count_fn(s.data) for s in sites],
        r_expected_total=fixture["expected"]["total"],
        r_expected_per_site=list(fixture["expected"]["per_site"]),
        query=QUERY,
        context=ctx,
        master=master,
    )


if __name__ == "__main__":  # pragma: no cover
    r = run()
    print(f"query            : {r.query}")
    print(f"per-site counts  : {r.per_site_cleartext}  (no party sees these)")
    print(f"encrypted total  : {r.total_encrypted}")
    print(f"R computed       : {r.r_expected_total}")
    print(f"exact agreement  : {r.matches_r}")
    print(
        f"sites holding a share: {len(r.master.sites)}; "
        f"master holds none: {not hasattr(r.master, 'secret_share')}"
    )
