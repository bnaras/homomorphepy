"""Backend-facade behavior — meaningful whether or not openfhe exists."""

from __future__ import annotations

import os
import sys

import pytest

from homomorphepy import _backend


def test_importing_the_package_does_not_import_openfhe():
    # The package must be usable (fixtures, data prep) where the
    # extension cannot be installed, which today includes macOS.
    import homomorphepy  # noqa: F401

    assert "openfhe" not in sys.modules or _backend.have_backend()


def test_missing_backend_explains_the_mislabelled_wheel():
    if _backend.have_backend():
        pytest.skip("openfhe is importable here")
    with pytest.raises(ImportError) as exc:
        _backend.backend()
    msg = str(exc.value)
    # The trap worth naming explicitly: pip reports success off
    # Linux/CPython-3.12 and the import fails afterwards.
    assert "py3-none-any" in msg or "mislabelled" in msg
    assert "1.5.1.0.24.4" in msg
    assert "docs/install.md" in msg


def test_thread_env_is_set_and_reported():
    prior = os.environ.pop("OMP_NUM_THREADS", None)
    try:
        ok = _backend.set_thread_env(2)
        assert os.environ["OMP_NUM_THREADS"] == "2"
        # Returns False once the extension is already loaded, because
        # OpenFHE latches its thread count at library load and the cap
        # can no longer be relied on.
        assert ok == ("openfhe" not in sys.modules)
    finally:
        if prior is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = prior


def test_thread_env_does_not_override_an_explicit_setting():
    prior = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "7"
    try:
        _backend.set_thread_env(2)
        assert os.environ["OMP_NUM_THREADS"] == "7"
    finally:
        if prior is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = prior


@pytest.mark.openfhe
def test_backend_round_trip():
    if not _backend.have_backend():
        pytest.skip("openfhe not installed")
    ofhe = _backend.backend()
    assert hasattr(ofhe, "CCParamsCKKSRNS")
