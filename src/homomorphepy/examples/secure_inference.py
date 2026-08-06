"""Secure inference: a lab scores encrypted patients.

Ported from homomorpheR's ``secure-inference.Rmd``. A hospital holds
patient biomarkers; a lab holds a proprietary linear risk model. The
hospital encrypts, the lab evaluates its model homomorphically on the
ciphertexts, and the hospital decrypts the scores.

Unlike the other examples in this tier, the data is packed in SIMD
layout: one ciphertext per biomarker, with the eight patients in the
slots. So the whole cohort is scored in four multiply-adds rather than
eight separate evaluations -- the arrangement that makes CKKS practical
for this shape of problem.

What the protocol does and does not give you
--------------------------------------------

It delivers two things: biomarker values never appear in cleartext
outside the hospital, and the lab's coefficients never reach the
hospital in cleartext. Both are necessary for a model-as-a-service
deployment that does not trust the lab with patient data.

Neither is sufficient. The hospital can query the lab repeatedly with
chosen inputs and recover the coefficients by solving the resulting
linear system -- for a linear model, five well-chosen queries suffice.
:func:`extract_model` demonstrates that, because a security claim that
is only ever stated is not a claim a reader can check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from homomorphepy.ciphertext import Ct
from homomorphepy.codec import packed_codec
from homomorphepy.context import Context, fhe_context
from homomorphepy.fixtures import load_json

__all__ = ["InferenceResult", "LAB_WEIGHTS", "LAB_BIAS", "run", "extract_model"]

# The lab's proprietary model. Never sent to the hospital.
LAB_WEIGHTS = (0.35, -0.20, 0.50, 0.15)
LAB_BIAS = 1.2


def _risk_band(score: float) -> str:
    if score > 2.0:
        return "HIGH"
    if score > 1.5:
        return "MODERATE"
    return "LOW"


@dataclass
class InferenceResult:
    scores_encrypted: list[float]
    scores_cleartext: list[float]
    bands: list[str]
    max_error: float
    n_patients: int
    context: Context = field(repr=False)


def run() -> InferenceResult:
    """Score the fixture cohort through the encrypted channel."""
    fixture = load_json("secure_inference")
    biomarkers = [fixture["biomarkers"][k] for k in ("b1", "b2", "b3", "b4")]
    n = len(biomarkers[0])

    # -- hospital: context, keys, and encryption ---------------------
    # Depth 2 covers the multiply plus the rescale the sum needs.
    ctx = fhe_context("CKKS", multiplicative_depth=2, scaling_mod_size=50, batch_size=8)
    keys = ctx.KeyGen()
    ctx.EvalMultKeyGen(keys.secretKey)
    codec = packed_codec(ctx)

    # One ciphertext per biomarker, patients across the slots.
    cts = [
        Ct(ctx.Encrypt(keys.publicKey, codec.encode(values)), ctx.cc)
        for values in biomarkers
    ]

    # -- lab: evaluate the model without decrypting ------------------
    # Reads exactly like the cleartext expression, which is the point
    # of the operator wrapper.
    score_ct = (
        cts[0] * LAB_WEIGHTS[0]
        + cts[1] * LAB_WEIGHTS[1]
        + cts[2] * LAB_WEIGHTS[2]
        + cts[3] * LAB_WEIGHTS[3]
        + LAB_BIAS
    )

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


def extract_model(n_queries: int = 5) -> dict[str, object]:
    """Recover the lab's coefficients from black-box query access.

    The attack the protocol does not prevent. The hospital submits
    chosen biomarker vectors, observes the returned scores, and solves
    the linear system. For a linear model with four weights and a bias,
    five linearly independent queries determine it exactly.

    Runs in the clear here: encryption is irrelevant to the attack,
    which is precisely the point. The lab's protection against it is
    rate limiting, query auditing, or a non-linear model -- not FHE.
    """
    rng = np.random.default_rng(20260805)
    X = rng.normal(size=(n_queries, len(LAB_WEIGHTS)))
    y = X @ np.array(LAB_WEIGHTS) + LAB_BIAS

    design = np.column_stack([X, np.ones(n_queries)])
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)

    recovered_w, recovered_b = solution[:-1], float(solution[-1])
    return {
        "n_queries": n_queries,
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
    print("  the coefficients are recoverable; FHE does not prevent this")
