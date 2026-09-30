"""The worked examples.

Each module exposes ``run()`` returning a result object, so the same
code serves the tests and the rendered documents.

**Simulated examples draw their own data.** Where the setting is a
data-generating process rather than a measurement, each run simulates
afresh: the claim being made is that the encrypted protocol reproduces
the cleartext answer *on whatever data it was given*, which is stronger
than reproducing one stored dataset.

**Measured data ships with the package.** The DLBCL cohort used by
:mod:`~homomorphepy.examples.cox`,
:mod:`~homomorphepy.examples.cox_lasso` and
:mod:`~homomorphepy.examples.dp` is what it is; there is no draw to
repeat, so every run reads the same bytes. See
:mod:`homomorphepy.fixtures`.

Ordered by dependency weight:

- :mod:`~homomorphepy.examples.aggregation` — encrypted counting under
  a single-decrypter aggregator (BFV).
- :mod:`~homomorphepy.examples.query_count` — the same count under
  threshold BFV, where no single party can decrypt.
- :mod:`~homomorphepy.examples.mle` — Poisson MLE driven through the
  encrypted channel by an unmodified optimizer.
- :mod:`~homomorphepy.examples.secure_inference` — a lab scoring
  encrypted patient data without seeing it.
- :mod:`~homomorphepy.examples.cox` — stratified Cox regression across
  sites, single-decrypter or threshold.
- :mod:`~homomorphepy.examples.cox_lasso` — the full pipeline on gene
  expression: pooled standardization, screening, and a penalized fit
  by consensus ADMM with an encrypted consensus step.
- :mod:`~homomorphepy.examples.dp` — what adding differential privacy
  on top costs, for the Cox fit and for consensus ADMM.

Submodules are not imported here: each pulls in the crypto backend,
and ``python -m homomorphepy.examples.mle`` would otherwise warn about
a double import. Import the one you want directly.
"""

__all__ = [
    "aggregation",
    "cox",
    "cox_lasso",
    "dp",
    "mle",
    "query_count",
    "secure_inference",
]
