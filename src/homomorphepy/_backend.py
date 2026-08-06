"""Typed facade over the ``openfhe`` extension module.

Two jobs:

1. Fail *usefully* when the extension is absent. The PyPI ``openfhe``
   wheels are tagged ``py3-none-any`` but contain
   ``cpython-312-x86_64-linux-gnu`` binaries, so pip reports a
   successful install on macOS and on CPython != 3.12 and the import
   then fails. Users hit that trap without this message.

2. Keep the rest of the package type-checkable. openfhe-python ships no
   type stubs, so everything crossing this boundary is ``Any``;
   confining the untyped surface to one module means pyright still has
   something to say about the protocol code.

Thread control note: openfhe-python binds no thread API at all, and
OpenFHE latches its machine-thread count when the shared library
loads. ``OMP_NUM_THREADS`` therefore has to be set *before* the first
import of this module — see :func:`set_thread_env`.
"""

from __future__ import annotations

import os
import sys
from typing import Any

__all__ = ["backend", "have_backend", "require_backend", "set_thread_env"]

_INSTALL_HELP = """\
homomorphepy needs the `openfhe` extension module, which is not importable.

  Linux x86_64, CPython 3.12 (Ubuntu 24.04):
      pip install 'openfhe==1.5.1.0.24.4'
      # Ubuntu 22.04 wants ...0.22.4 instead. The Ubuntu release is
      # encoded in the VERSION, not the wheel tag, so never pin with
      # a wildcard: 'openfhe==1.5.1.0.*' resolves to the 24.04 build
      # on both.

  macOS, Windows, or any other CPython:
      No wheel exists. Build openfhe-python from source against a
      local OpenFHE and install the resulting wheel. See docs/install.md.

  Already ran pip successfully and still see this?
      That is the expected failure mode off Linux/CPython-3.12: the
      wheels are mislabelled `py3-none-any`, so pip installs them
      anywhere and the import fails afterwards.
"""


def set_thread_env(n: int = 2) -> bool:
    """Cap OpenMP threads. Only effective *before* the first import.

    OpenFHE's ``OpenFHEParallelControls`` latches ``omp_get_max_threads()``
    in its constructor, which runs when the extension's shared library
    loads, and the library's hot regions carry explicit ``num_threads``
    clauses that override the OpenMP thread-count variable afterwards.
    So a cap applied after import does not reliably take effect, and
    threadpoolctl does not help either.

    Call this at the top of a module or notebook, before importing
    anything that imports ``openfhe``; in CI, prefer setting the
    variable in the job environment. Upstream ``openfhe-python`` binds
    no ``SetNumThreads``, which is filed as a defect; when a release
    exposes it this function should switch to calling it.

    Returns True if the variable was set, False if the extension was
    already loaded (in which case the cap may not hold).
    """
    already = "openfhe" in sys.modules
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    return not already


def have_backend() -> bool:
    """True if the extension module can be imported."""
    try:
        import openfhe  # noqa: F401
    except ImportError:
        return False
    return True


def backend() -> Any:
    """The ``openfhe`` module, or raise with install guidance."""
    try:
        import openfhe
    except ImportError as exc:
        raise ImportError(_INSTALL_HELP) from exc
    return openfhe


def require_backend() -> Any:
    """Alias for :func:`backend`, reading better at call sites."""
    return backend()
