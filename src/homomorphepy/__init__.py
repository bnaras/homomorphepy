"""homomorphepy — multi-site privacy-preserving statistics over FHE.

The Python twin of the R package homomorpheR, built on openfhe-python.
Both packages sit on the same OpenFHE C++ library, and their worked
examples consume the same fixture bytes so results can be compared
across languages rather than merely resembling one another.

Importing this package does NOT import the crypto backend; see
`homomorphepy._backend` for why that is deliberate and how to install
it.
"""

from homomorphepy._backend import backend, have_backend, set_thread_env
from homomorphepy.fixtures import (
    FixtureError,
    Tolerance,
    fixture_dir,
    load_dlbcl,
    load_dlbcl_gex,
    load_golden,
    load_json,
    manifest,
    site_order,
    verify_all,
)

__version__ = "1.5.1.dev0"

__all__ = [
    "FixtureError",
    "Tolerance",
    "__version__",
    "backend",
    "fixture_dir",
    "have_backend",
    "load_dlbcl",
    "load_dlbcl_gex",
    "load_golden",
    "load_json",
    "manifest",
    "set_thread_env",
    "site_order",
    "verify_all",
]
