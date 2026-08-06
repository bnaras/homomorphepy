"""Privacy-preserving aggregation: encrypted counts across sites.

Ported from homomorpheR's ``privacy-preserving-aggregation.Rmd``. The
simplest composition with FHE, and the one to read first.

Three sites each hold a patient cohort. Each counts locally how many of
its patients match a query, encrypts that single integer under the
coordinator's public key, and sends the ciphertext. The coordinator
adds the ciphertexts homomorphically and decrypts only the total. It
never sees a per-site count.

BFV is used rather than CKKS because counts are integers and BFV is
exact: the recovered total is the integer, not an approximation of it.

This example deliberately keeps the coordinator as a single decrypter
-- it holds the secret key and could in principle decrypt an individual
site's ciphertext if one were sent alone. :mod:`.query_count` runs the
same computation under threshold keys, where no single party can
decrypt at all; the contrast between the two is the point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from homomorphepy.ciphertext import Ct
from homomorphepy.codec import packed_codec
from homomorphepy.context import Context, fhe_context
from homomorphepy.fixtures import load_json

__all__ = ["AggregationResult", "run"]

PLAINTEXT_MODULUS = 65537


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


def site_count(rows: dict[str, list[Any]]) -> int:
    """The query each site evaluates locally, on its own data."""
    return sum(
        1
        for i in range(len(rows["age"]))
        if rows["age"][i] < 50 and rows["sex"][i] == "F" and rows["biomarker"][i] < 0.2
    )


def run() -> AggregationResult:
    """Run the protocol over the R-exported cohorts."""
    fixture = load_json("aggregation_sites")
    sites = fixture["sites"]

    # The coordinator sets up the context and keys, and distributes the
    # public key. In a deployment the serialized context travels too;
    # here one process stands in for all parties.
    ctx = fhe_context(
        "BFV", plaintext_modulus=PLAINTEXT_MODULUS, multiplicative_depth=1
    )
    keys = ctx.KeyGen()
    codec = packed_codec(ctx)

    # Each site encrypts one integer under the coordinator's public key.
    per_site = [site_count(rows) for rows in sites]
    ciphertexts = [
        Ct(ctx.Encrypt(keys.publicKey, codec.encode(count)), ctx.cc)
        for count in per_site
    ]

    # The coordinator adds without decrypting anything intermediate.
    total_ct = sum(ciphertexts)

    pt = ctx.Decrypt(total_ct.raw, keys.secretKey)
    total = codec.decode(pt, 1)[0]

    return AggregationResult(
        total_encrypted=int(total),
        total_cleartext=sum(per_site),
        per_site_cleartext=per_site,
        site_sizes=[len(r["age"]) for r in sites],
        context=ctx,
    )


if __name__ == "__main__":  # pragma: no cover
    r = run()
    print(f"site sizes        : {r.site_sizes}")
    print(
        f"per-site counts   : {r.per_site_cleartext}  (never seen by the coordinator)"
    )
    print(f"encrypted total   : {r.total_encrypted}")
    print(f"cleartext total   : {r.total_cleartext}")
    print(f"exact             : {r.exact}")
