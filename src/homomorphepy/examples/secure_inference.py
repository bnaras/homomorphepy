"""Secure inference: a lab scores encrypted patients.

A hospital holds
patient biomarkers; a lab holds a proprietary linear risk model. The
hospital encrypts, the lab evaluates its model homomorphically on the
encrypted values, and the hospital decrypts the scores.

Unlike the other examples in this tier, the data is packed across
slots: one encrypted value per biomarker, with the eight patients in the
slots. So the whole cohort is scored in four multiply-adds rather than
eight separate evaluations -- the arrangement that makes CKKS practical
for this shape of problem.

What the protocol does and does not give you
--------------------------------------------

It delivers two things: biomarker values never appear in cleartext
outside the hospital, and the lab's coefficients never reach the
hospital in cleartext. Both are necessary for a model-as-a-service
deployment that does not trust the lab with patient data.

Neither is sufficient. The hospital decides what it encrypts, so it can
submit the standard basis as queries: for a linear model with four
biomarkers and a bias, five queries recover every coefficient.
:func:`extract_model` runs that attack through the encrypted pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from homomorphepy.ciphertext import Ct
from homomorphepy.codec import packed_codec
from homomorphepy.context import Context, fhe_context

__all__ = [
    "InferenceResult",
    "BIOMARKERS",
    "LAB_WEIGHTS",
    "LAB_BIAS",
    "run",
    "extract_model",
]

# Eight patients, four biomarkers each. A fixed panel rather than a
# random draw: the example is about the pipeline, and fixed values keep
# the printed scores stable between runs.
BIOMARKERS = (
    (1.2, 0.8, 1.5, 0.3, 2.1, 0.9, 1.1, 1.8),
    (0.5, 1.1, 0.3, 0.8, 0.2, 1.4, 0.7, 0.6),
    (2.0, 1.5, 2.3, 1.0, 1.8, 2.1, 1.6, 2.5),
    (0.1, 0.4, 0.2, 0.6, 0.3, 0.1, 0.5, 0.2),
)

# The lab's proprietary model. Never sent to the hospital.
LAB_WEIGHTS = (0.35, -0.20, 0.50, 0.15)
LAB_BIAS = 1.2


def _risk_band(score: float) -> str:
    if score > 2.0:
        return "HIGH"
    if score > 1.5:
        return "MODERATE"
    return "LOW"


def _hospital_context():
    """The hospital's CKKS context, key pair and packing codec."""
    # Depth 2 covers the multiply plus the rescale the sum needs.
    ctx = fhe_context("CKKS", multiplicative_depth=2, scaling_mod_size=50, batch_size=8)
    keys = ctx.KeyGen()
    # Keys for multiplying two encrypted values. The scoring below
    # multiplies encrypted values only by the lab's cleartext weights,
    # which does not use them.
    ctx.EvalMultKeyGen(keys.secretKey)
    return ctx, keys, packed_codec(ctx)


def _lab_score(cts: list[Ct]) -> Ct:
    """The lab's model applied to four encrypted biomarker vectors.

    Reads exactly like the cleartext expression. The weights stay in
    this function; the caller never reads them.
    """
    return (
        cts[0] * LAB_WEIGHTS[0]
        + cts[1] * LAB_WEIGHTS[1]
        + cts[2] * LAB_WEIGHTS[2]
        + cts[3] * LAB_WEIGHTS[3]
        + LAB_BIAS
    )


@dataclass
class InferenceResult:
    scores_encrypted: list[float]
    scores_cleartext: list[float]
    bands: list[str]
    max_error: float
    n_patients: int
    context: Context = field(repr=False)


def run() -> InferenceResult:
    """Score the biomarker panel through the encrypted channel."""
    biomarkers = [list(b) for b in BIOMARKERS]
    n = len(biomarkers[0])

    # -- hospital: context, keys, and encryption ---------------------
    ctx, keys, codec = _hospital_context()

    # One encrypted value per biomarker, patients across the slots.
    cts = [
        Ct(ctx.Encrypt(keys.publicKey, codec.encode(values)), ctx.cc)
        for values in biomarkers
    ]

    # -- lab: evaluate the model without decrypting ------------------
    score_ct = _lab_score(cts)

    # -- hospital: decrypt -------------------------------------------
    pt = ctx.Decrypt(score_ct.raw, keys.secretKey)
    scores = codec.decode(pt, n)

    reference = (np.array(LAB_WEIGHTS) @ np.array(biomarkers) + LAB_BIAS).tolist()

    return InferenceResult(
        scores_encrypted=[float(s) for s in scores],
        scores_cleartext=[float(s) for s in reference],
        bands=[_risk_band(float(s)) for s in scores],
        max_error=float(np.max(np.abs(np.array(scores) - np.array(reference)))),
        n_patients=n,
        context=ctx,
    )


def extract_model() -> dict[str, object]:
    """Recover the lab's coefficients through the encrypted pipeline.

    The attack the protocol does not prevent. The hospital encrypts the
    standard basis as five probes packed across slots 1-5 of the four
    biomarker vectors: slot 1 is all zeros and returns ``b``; slot
    ``j + 1`` has a one in position ``j`` and returns ``w_j + b``. The
    lab scores them as it would any patients, and the hospital
    decrypts. Subtracting the first score from the others recovers the
    weights. A linear function of ``k`` inputs is determined by any
    ``k + 1`` affinely independent evaluations, so five queries
    suffice.
    """
    ctx, keys, codec = _hospital_context()
    k = len(LAB_WEIGHTS)
    probes = np.zeros((k, k + 1))
    probes[np.arange(k), np.arange(1, k + 1)] = 1.0

    cts = [
        Ct(ctx.Encrypt(keys.publicKey, codec.encode(row.tolist())), ctx.cc)
        for row in probes
    ]
    pt = ctx.Decrypt(_lab_score(cts).raw, keys.secretKey)
    scores = np.asarray(codec.decode(pt, k + 1), dtype=float)

    recovered_b = float(scores[0])
    recovered_w = scores[1:] - recovered_b
    return {
        "n_queries": k + 1,
        "recovered_weights": recovered_w.tolist(),
        "recovered_bias": recovered_b,
        "max_weight_error": float(np.max(np.abs(recovered_w - np.array(LAB_WEIGHTS)))),
        "bias_error": abs(recovered_b - LAB_BIAS),
    }


if __name__ == "__main__":  # pragma: no cover
    r = run()
    print(f"patients scored : {r.n_patients}")
    for i, (s, band) in enumerate(zip(r.scores_encrypted, r.bands, strict=True), 1):
        print(f"  patient {i}: {s:.3f} ({band})")
    print(f"max error vs cleartext : {r.max_error:.2e}")
    print()
    a = extract_model()
    print(f"model extraction with {a['n_queries']} queries:")
    print(f"  max weight error : {a['max_weight_error']:.2e}")
    print(f"  bias error       : {a['bias_error']:.2e}")
    print("  the coefficients are recoverable through the encrypted pipeline")
