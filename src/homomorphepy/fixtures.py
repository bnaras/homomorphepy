"""Loading of the shipped datasets.

The examples built on measured data -- the Rosenwald DLBCL cohort and
its expression matrix -- read it from here, so every run computes on
the same bytes. Simulated examples do not use this module at all; they
draw their own data.

The expression matrix ships as raw float64 rather than text. At 15
significant digits a CSV round-trip drops bits that matter: the
screening step in :mod:`~homomorphepy.examples.cox_lasso` ranks 6416
probes and keeps 100, and probes nearly tied at that boundary can swap
under a one-ulp perturbation.

Everything in this module asserts rather than infers. The manifest
declares each column's dtype and each factor's category order, and the
loaders apply those declarations; nothing is left to pandas' type
inference or to alphabetical sorting.

Also here: :func:`load_golden` and :func:`load_json`, which back the
project's own parity checks in ``tests/``. They are not needed to use
the package.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "FixtureError",
    "fixture_dir",
    "manifest",
    "load_json",
    "load_dlbcl",
    "load_dlbcl_gex",
    "load_golden",
    "verify_all",
    "Tolerance",
]


class FixtureError(RuntimeError):
    """A fixture is missing, stale, or does not match its declaration."""


def fixture_dir() -> Path:
    """Directory holding the parity fixtures.

    Prefers the copy shipped inside the installed package; falls back to
    the monorepo's ``fixtures/parity`` when running from a source
    checkout that has not been staged yet.
    """
    shipped = Path(__file__).parent / "fixtures"
    if (shipped / "manifest.json").exists():
        return shipped

    # Source checkout: py_dev/homomorphepy/src/homomorphepy/ -> repo root
    monorepo = Path(__file__).resolve().parents[4] / "fixtures" / "parity"
    if (monorepo / "manifest.json").exists():
        return monorepo

    raise FixtureError(
        "No parity fixtures found. Expected either\n"
        f"  {shipped}\n"
        f"or {monorepo}\n"
        "Stage them with: bash fixtures/sync_fixtures.sh "
        "py_dev/homomorphepy/src/homomorphepy/fixtures"
    )


@lru_cache(maxsize=1)
def manifest() -> dict[str, Any]:
    """The fixture manifest, keyed for lookup by filename."""
    d = fixture_dir()
    man = json.loads((d / "manifest.json").read_text())
    man["_entries"] = {f["file"]: f for f in man["files"]}
    man["_dir"] = str(d)
    return man


def _entry(name: str) -> dict[str, Any]:
    try:
        return manifest()["_entries"][name]
    except KeyError:
        raise FixtureError(f"{name} is not declared in the manifest") from None


def _checked_bytes(name: str) -> bytes:
    """Read a fixture, failing loudly if its digest does not match.

    A stale fixture is worse than a missing one: it turns a data
    problem into what looks like a binding defect.
    """
    e = _entry(name)
    path = Path(manifest()["_dir"]) / name
    if not path.exists():
        raise FixtureError(f"missing fixture: {path}")
    raw = path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != e["sha256"]:
        raise FixtureError(
            f"{name} does not match its manifest digest "
            f"(expected {e['sha256'][:12]}..., got {got[:12]}...). "
            "Regenerate with fixtures/sync_fixtures.sh"
        )
    return raw


def load_json(name: str) -> dict[str, Any]:
    """Load and digest-check one of the JSON fixtures."""
    if not name.endswith(".json"):
        name = f"{name}.json"
    return json.loads(_checked_bytes(name).decode())


def _apply_dtypes(raw: pd.DataFrame, dtypes: dict[str, Any]) -> pd.DataFrame:
    """Apply the manifest's declared dtypes to an all-string frame.

    Categoricals get their declared level order, which is protocol semantics
    rather than presentation: sites are visited in level order and the
    first is the lead decryptor in the threshold decryption. Letting
    pandas sort them alphabetically would silently permute the protocol.
    """
    out = pd.DataFrame(index=raw.index)
    for col, spec in dtypes.items():
        if col not in raw.columns:
            raise FixtureError(f"declared column {col!r} absent from the data")
        if isinstance(spec, dict) and spec.get("dtype") == "category":
            out[col] = pd.Categorical(
                raw[col],
                categories=list(spec["categories"]),
                ordered=bool(spec.get("ordered", True)),
            )
            if out[col].isna().any():
                bad = sorted(set(raw[col]) - set(spec["categories"]))
                raise FixtureError(
                    f"{col!r} has values outside its declared categories: {bad}"
                )
        elif spec == "int64":
            out[col] = raw[col].astype("int64")
        elif spec == "float64":
            out[col] = raw[col].astype("float64")
        elif spec == "str":
            out[col] = raw[col].astype("string")
        else:
            raise FixtureError(f"unknown dtype declaration for {col!r}: {spec!r}")
    return out


def load_dlbcl() -> pd.DataFrame:
    """The DLBCL clinical table, with dtypes and site order forced."""
    name = "dlbcl_clinical.csv"
    e = _entry(name)
    raw_bytes = _checked_bytes(name)
    from io import BytesIO

    raw = pd.read_csv(BytesIO(raw_bytes), dtype=str)
    df = _apply_dtypes(raw, e["dtypes"])

    if df.shape != (e["nrow"], e["ncol"]):
        raise FixtureError(f"expected shape {(e['nrow'], e['ncol'])}, got {df.shape}")

    observed = {c: int((df["Subgroup"] == c).sum()) for c in site_order()}
    declared = {k: int(v) for k, v in e["site_sizes"].items()}
    if observed != declared:
        raise FixtureError(f"site sizes {observed} != declared {declared}")
    return df


def site_order() -> list[str]:
    """Canonical site order: index 0 is the lead decryptor."""
    return list(_entry("dlbcl_clinical.csv")["site_order"])


def load_dlbcl_gex() -> tuple[np.ndarray, list[str], list[str]]:
    """The expression matrix as float64, with its row and column names.

    Stored as gzipped raw little-endian float64 rather than CSV: the
    Cox-lasso screen ranks 6416 probes and keeps the top 100, and
    near-ties at that boundary flip under a one-ulp perturbation, which
    text formatting would introduce.
    """
    name = "dlbcl_gex.f64.gz"
    e = _entry(name)
    blob = gzip.decompress(_checked_bytes(name))

    if "sha256_content" in e:
        got = hashlib.sha256(blob).hexdigest()
        if got != e["sha256_content"]:
            raise FixtureError(
                f"{name}: uncompressed content digest mismatch "
                f"(expected {e['sha256_content'][:12]}..., got {got[:12]}...)"
            )

    nrow, ncol = e["shape"]
    gex = np.frombuffer(blob, dtype="<f8").reshape(nrow, ncol)

    d = Path(manifest()["_dir"])
    rows = (d / e["rownames_file"]).read_text().split()
    cols = (d / e["colnames_file"]).read_text().split()
    if len(rows) != nrow or len(cols) != ncol:
        raise FixtureError(
            f"dimnames {len(rows)}x{len(cols)} disagree with shape {nrow}x{ncol}"
        )
    return gex, rows, cols


@dataclass(frozen=True)
class Tolerance:
    """One rung of the comparison ladder.

    ``kind`` distinguishes what a difference would *mean*:

    ``ckks``
        within-language encrypted vs plaintext — the actual
        cryptographic claim, and the tightest meaningful bound.
    ``statistical``
        cross-language, where CVXR and cvxpy canonicalize
        independently and ship different solver builds. Orders of
        magnitude looser than ``ckks``; conflating the two is how a
        solver difference gets misread as crypto noise.
    ``iteration_count``
        an integer slack on a loop whose stopping rule is absolute.
    ``set_equality``
        no slack at all — deterministic given identical input bytes.
    """

    name: str
    value: float
    kind: str
    note: str = ""

    def holds(self, diff: float) -> bool:
        return float(diff) <= self.value


def load_golden(name: str = "cvxr_consensus_golden") -> dict[str, Any]:
    """A golden-output fixture, with its tolerances parsed."""
    g = load_json(name)
    g["tolerances"] = {
        k: Tolerance(name=k, value=v["value"], kind=v["kind"], note=v.get("note", ""))
        for k, v in g.get("tolerances", {}).items()
    }
    return g


def verify_all() -> dict[str, str]:
    """Digest-check every declared fixture. Returns {file: sha256}."""
    out = {}
    for name in manifest()["_entries"]:
        out[name] = hashlib.sha256(_checked_bytes(name)).hexdigest()
    return out
