"""Worked examples, ported from the homomorpheR vignettes.

Each module exposes ``run()`` returning a result object, so the same
code serves the tests and the rendered documents. Inputs come from the
R-exported fixtures rather than being re-simulated, because R's RNG has
no numpy equivalent -- see :mod:`homomorphepy.fixtures`.

Ordered by dependency weight, as in the port plan:

- :mod:`~homomorphepy.examples.aggregation` — encrypted counting under
  a single-decrypter coordinator (BFV).
- :mod:`~homomorphepy.examples.query_count` — the same count under
  threshold BFV, where no single party can decrypt.
- :mod:`~homomorphepy.examples.mle` — Poisson MLE driven through the
  encrypted channel by an unmodified optimizer.
- :mod:`~homomorphepy.examples.secure_inference` — a lab scoring
  encrypted patient data without seeing it.


Submodules are not imported here: each pulls in the crypto backend,
and ``python -m homomorphepy.examples.mle`` would otherwise warn about
a double import. Import the one you want directly.
"""

__all__ = ["aggregation", "mle", "query_count", "secure_inference"]
