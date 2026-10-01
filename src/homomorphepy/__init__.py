"""homomorphepy — multi-site privacy-preserving statistics over FHE.

Multi-site privacy-preserving statistics over homomorphic encryption,
built on openfhe-python.
Both packages sit on the same OpenFHE C++ library, and their worked
examples consume the same fixture bytes so results can be compared
across languages rather than merely resembling one another.

Importing this package does NOT import the crypto backend; see
`homomorphepy._backend` for why that is deliberate and how to install
it.
"""

from homomorphepy._backend import backend, have_backend, set_thread_env
from homomorphepy.actors import (
    CKKSMaster,
    Master,
    RemoteSite,
    Site,
    SiteUnavailable,
    ThresholdMaster,
    make_ckks_master,
    make_joint_rotation_keys,
    make_site,
    make_threshold_master,
    make_worker,
)
from homomorphepy.ciphertext import Ct, unwrap, wrap
from homomorphepy.codec import Codec, as_exact_integer, packed_codec
from homomorphepy.context import Context, Scheme, fhe_context
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
from homomorphepy.params import (
    BadContribution,
    KeyMismatch,
    OpenFHEParams,
    PublicParams,
)

__version__ = "1.0"

__all__ = [
    "make_worker",
    "make_threshold_master",
    "make_site",
    "make_ckks_master",
    "make_joint_rotation_keys",
    "ThresholdMaster",
    "RemoteSite",
    "Site",
    "SiteUnavailable",
    "Master",
    "BadContribution",
    "CKKSMaster",
    "Codec",
    "Context",
    "Ct",
    "FixtureError",
    "KeyMismatch",
    "OpenFHEParams",
    "PublicParams",
    "Scheme",
    "Tolerance",
    "__version__",
    "as_exact_integer",
    "backend",
    "fhe_context",
    "fixture_dir",
    "have_backend",
    "load_dlbcl",
    "load_dlbcl_gex",
    "load_golden",
    "load_json",
    "manifest",
    "packed_codec",
    "set_thread_env",
    "site_order",
    "unwrap",
    "verify_all",
    "wrap",
]
