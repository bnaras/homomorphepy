"""Privacy-preserving aggregation: encrypted counts across sites.

The simplest composition with FHE, and the one to read first.

Three sites each hold a patient cohort. Each counts locally how many of
its patients match a query, encrypts that single integer under the
aggregator's public key, and sends the encrypted count. The aggregator
adds those encrypted counts and decrypts only the total. It
never sees a per-site count.

Unlike the other examples this one calls the encryption primitives
directly -- context, key generation, encode, encrypt, add, decrypt --
rather than going through the actor layer, because the primitives are
what it is teaching.

BFV is used rather than CKKS because counts are integers and BFV is
exact: the recovered total is the integer, not an approximation of it.

The aggregator here is a single decrypter -- it holds the secret key
and could in principle decrypt an individual site's contribution if one
were sent alone. :mod:`.query_count` runs the same computation under
threshold keys, where no single party can decrypt at all; the contrast
between the two is the point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from homomorphepy.ciphertext import Ct
from homomorphepy.codec import packed_codec
from homomorphepy.context import Context, fhe_context

__all__ = ["AggregationResult", "QUERY", "SITE_SIZES", "simulate", "site_count", "run"]

PLAINTEXT_MODULUS = 65537
SITE_SIZES = (1000, 500, 1500)
QUERY = 'age < 50 and sex == "F" and biomarker < 0.2'


def simulate(seed: int = 42) -> list[pd.DataFrame]:
    """Three site cohorts: age, sex, and a biomarker on [0, 1]."""
    rng = np.random.default_rng(seed)
    return [
        pd.DataFrame(
            {
                "age": rng.integers(40, 71, size=n),
                "sex": rng.choice(["M", "F"], size=n),
                "biomarker": rng.uniform(0.0, 1.0, size=n),
            }
        )
        for n in SITE_SIZES
    ]


def site_count(data: pd.DataFrame, query: str = QUERY) -> int:
    """The query each site evaluates locally, on its own data."""
    return int(len(data.query(query)))


@dataclass
class AggregationResult:
    total_encrypted: int
    total_cleartext: int
    per_site_cleartext: list[int]
    site_sizes: list[int]
    context: Context

    @property
    def exact(self) -> bool:
        """BFV is exact arithmetic; this must always be True."""
        return self.total_encrypted == self.total_cleartext


def run(seed: int = 42, query: str = QUERY) -> AggregationResult:
    """Run the single-decrypter aggregation over simulated cohorts."""
    site_data = simulate(seed)

    # The aggregator sets up the context and keys, and distributes the
    # public key. In a deployment the serialized context travels too;
    # here one process stands in for all parties.
    ctx = fhe_context(
        "BFV", plaintext_modulus=PLAINTEXT_MODULUS, multiplicative_depth=1
    )
    keys = ctx.KeyGen()
    codec = packed_codec(ctx)

    # Each site encrypts one integer under the aggregator's public key.
    per_site = [site_count(rows, query) for rows in site_data]
    ciphertexts = [
        Ct(ctx.Encrypt(keys.publicKey, codec.encode(count)), ctx.cc)
        for count in per_site
    ]

    # The aggregator adds without decrypting anything intermediate.
    total_ct = sum(ciphertexts)

    pt = ctx.Decrypt(total_ct.raw, keys.secretKey)
    total = codec.decode(pt, 1)[0]

    return AggregationResult(
        total_encrypted=int(total),
        total_cleartext=sum(per_site),
        per_site_cleartext=per_site,
        site_sizes=[len(r) for r in site_data],
        context=ctx,
    )


if __name__ == "__main__":  # pragma: no cover
    r = run()
    print(f"site sizes        : {r.site_sizes}")
    print(f"per-site counts   : {r.per_site_cleartext}  (never seen by the aggregator)")
    print(f"encrypted total   : {r.total_encrypted}")
    print(f"cleartext total   : {r.total_cleartext}")
    print(f"exact             : {r.exact}")
