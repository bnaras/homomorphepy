#!/usr/bin/env bash
# build-openfhe-wheel.sh — Build openfhe-python from source and pack a
# proper, installable wheel.
#
# WHY THIS EXISTS (upstream defect PKG2)
#
# openfhe-python ships no setup.py and no pyproject.toml. Its CMake
# copies the extension module and src/__init__.py straight into
# site-packages with NO dist-info, so pip and uv cannot see it: the
# resolver does not know it is installed, and `uv sync` -- which prunes
# to the locked set -- is free to remove it or rebuild the environment
# without it. The upstream pip path lives in a separate Linux-only
# repo. On macOS there is no supported install route at all.
#
# So we build the module and pack a real wheel around it: correct
# platform tag, real dist-info, the OpenFHE dylibs vendored alongside,
# and rpaths rewritten so nothing depends on absolute build paths.
#
# WHY WE DO NOT JUST USE THE PyPI WHEEL (upstream defect PKG1)
#
# The published wheels are tagged py3-none-any but contain
# cpython-312-x86_64-linux-gnu binaries. They install "successfully" on
# macOS and then fail at import. There is no macOS wheel.
#
# Usage:
#   OPENFHE_PYTHON_SRC=/path/to/openfhe-python \
#   OPENFHE_HOME=/path/to/openfhe/install \
#     bash scripts/build-openfhe-wheel.sh [--python 3.12] [--jobs N] [--clean]
#
# OPENFHE_PYTHON_SRC is an openfhe-python source checkout; OPENFHE_HOME
# is the OpenFHE C++ install prefix it builds against (the directory
# holding lib/OpenFHE). Build scratch goes under build/ in this repo.
#
# Output: dist/openfhe-<ver>-cp3XX-cp3XX-macosx_<arch>.whl

set -euo pipefail

PYVER="3.12"
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
CLEAN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PYVER="$2"; shift 2 ;;
        --jobs)   JOBS="$2";  shift 2 ;;
        --clean)  CLEAN=true; shift ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
done

: "${OPENFHE_PYTHON_SRC:?set OPENFHE_PYTHON_SRC to an openfhe-python source checkout}"
: "${OPENFHE_HOME:?set OPENFHE_HOME to the OpenFHE install prefix (holding lib/OpenFHE)}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$OPENFHE_PYTHON_SRC"
OPENFHE_PREFIX="$OPENFHE_HOME"
SCRATCH="$HERE/build"
BUILD="$SCRATCH/openfhe-python-build"
STAGE="$SCRATCH/openfhe-wheel-stage"
DIST="$HERE/dist"

for d in "$SRC" "$OPENFHE_PREFIX/lib/OpenFHE"; do
    [[ -d "$d" ]] || { echo "ERROR: missing $d" >&2; exit 1; }
done
mkdir -p "$SCRATCH"

echo "== configuration =="
echo "  source      : $SRC"
echo "  OpenFHE     : $OPENFHE_PREFIX"
echo "  python      : $PYVER"
echo

# Interpreter and pybind11 both have to come from the SAME environment,
# or pybind11_add_module builds against the wrong Python ABI.
echo "== provisioning build environment =="
BUILD_VENV="$SCRATCH/openfhe-build-venv"
uv venv --python "$PYVER" "$BUILD_VENV" --quiet --allow-existing
VENV_PY="$BUILD_VENV/bin/python"
uv pip install --python "$VENV_PY" --quiet pybind11 wheel
PYBIND_CMAKE="$("$VENV_PY" -c 'import pybind11; print(pybind11.get_cmake_dir())')"
ABI_TAG="$("$VENV_PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
echo "  pybind11    : $PYBIND_CMAKE"
echo "  ext suffix  : $ABI_TAG"
echo

# -- Build ------------------------------------------------------------------
echo "== building =="
# Incremental by default; --clean forces a fresh configure+compile.
$CLEAN && rm -rf "$BUILD"
cmake -S "$SRC" -B "$BUILD" \
      -DCMAKE_BUILD_TYPE=Release \
      -DOpenFHE_DIR="$OPENFHE_PREFIX/lib/OpenFHE" \
      -DCMAKE_PREFIX_PATH="$PYBIND_CMAKE" \
      -DPython_EXECUTABLE="$VENV_PY" \
      -DPYTHON_EXECUTABLE_PATH="$VENV_PY" \
      > "$SCRATCH/openfhe-python-build.configure.log" 2>&1 \
  || { echo "CONFIGURE FAILED — see build/openfhe-python-build.configure.log" >&2
       tail -25 "$SCRATCH/openfhe-python-build.configure.log" >&2; exit 1; }

cmake --build "$BUILD" -j "$JOBS" \
      > "$SCRATCH/openfhe-python-build.build.log" 2>&1 \
  || { echo "BUILD FAILED — see build/openfhe-python-build.build.log" >&2
       tail -25 "$SCRATCH/openfhe-python-build.build.log" >&2; exit 1; }

MODULE="$(find "$BUILD" -maxdepth 1 -name "openfhe*.so" | head -1)"
[[ -n "$MODULE" ]] || { echo "ERROR: no extension module produced" >&2; exit 1; }
echo "  built: $(basename "$MODULE")"
echo

# -- Stage the wheel tree ---------------------------------------------------
# Layout mirrors the published wheel: a package dir with the extension
# and a lib/ subdirectory that @loader_path/lib resolves against.
echo "== staging =="
rm -rf "$STAGE"; mkdir -p "$STAGE/openfhe/lib"
cp "$MODULE" "$STAGE/openfhe/"
cp "$SRC/src/__init__.py" "$STAGE/openfhe/"
for lib in "$OPENFHE_PREFIX"/lib/libOPENFHE*.dylib; do
    # Copy real files only; the *.1.dylib / *.dylib symlinks would
    # otherwise triple the wheel size.
    [[ -L "$lib" ]] && continue
    cp "$lib" "$STAGE/openfhe/lib/"
done

# Vendor libomp too. The OpenFHE build links R's copy by absolute,
# R-VERSION-PINNED path -- which is exactly what broke the previous
# reference venv when R moved 4.5 -> 4.6 and the old path vanished.
# Copying it in and rewriting the references to @loader_path makes the
# wheel independent of the R installation.
LIBOMP_SRC="$(otool -L "$STAGE/openfhe/lib/libOPENFHEcore.1.5.1.dylib" \
              | awk '/libomp\.dylib/ {print $1}' | head -1)"
if [[ -n "$LIBOMP_SRC" && -f "$LIBOMP_SRC" ]]; then
    cp "$LIBOMP_SRC" "$STAGE/openfhe/lib/libomp.dylib"
    chmod u+w "$STAGE/openfhe/lib/libomp.dylib"
    install_name_tool -id "@loader_path/libomp.dylib" \
        "$STAGE/openfhe/lib/libomp.dylib" 2>/dev/null || true
    echo "  vendored libomp from $LIBOMP_SRC"
else
    echo "  WARNING: could not locate libomp to vendor ($LIBOMP_SRC)" >&2
fi

echo "== rewriting install names =="
# Only real files were copied, not the libFOO.1.dylib -> libFOO.1.5.1.dylib
# symlinks (zip stores symlinks as plain files and pip does not restore
# them, so shipping them would produce broken duplicates). Dependencies
# are recorded against the SONAME, though, so every reference has to be
# resolved to the file that is actually present.
resolve_dep() {
    local dep_base="$1" prefix
    if [[ -f "$STAGE/openfhe/lib/$dep_base" ]]; then
        echo "$dep_base"; return
    fi
    # libOPENFHEpke.1.dylib -> libOPENFHEpke. -> libOPENFHEpke.1.5.1.dylib
    prefix="${dep_base%%.*}."
    local found
    found="$(cd "$STAGE/openfhe/lib" && ls "${prefix}"*.dylib 2>/dev/null | head -1)"
    echo "${found:-$dep_base}"
}

for lib in "$STAGE/openfhe/lib"/*.dylib; do
    chmod u+w "$lib"
    base="$(basename "$lib")"
    install_name_tool -id "@loader_path/$base" "$lib" 2>/dev/null || true
    # Point every OpenFHE->OpenFHE and OpenFHE->libomp reference at a
    # path relative to the loading object, so nothing resolves through
    # an absolute build-tree or R-version path.
    for dep in $(otool -L "$lib" | tail -n +2 | awk '{print $1}'); do
        case "$dep" in
            */libOPENFHE*|*libomp.dylib)
                install_name_tool -change "$dep" \
                    "@loader_path/$(resolve_dep "$(basename "$dep")")" \
                    "$lib" 2>/dev/null || true ;;
        esac
    done
done
chmod u+w "$STAGE/openfhe/$(basename "$MODULE")"
for dep in $(otool -L "$STAGE/openfhe/$(basename "$MODULE")" | tail -n +2 | awk '{print $1}'); do
    case "$dep" in
        */libOPENFHE*|*libomp.dylib)
            install_name_tool -change "$dep" \
                "@loader_path/lib/$(resolve_dep "$(basename "$dep")")" \
                "$STAGE/openfhe/$(basename "$MODULE")" 2>/dev/null || true ;;
    esac
done

# Dependency lines only: otool also echoes each file's own path as a
# header, which would otherwise always "match" the staging directory.
# Re-sign. On Apple Silicon every Mach-O must carry a valid signature,
# and install_name_tool invalidates the one it rewrites -- the loader
# then SIGKILLs the process (exit 137) at import, with no diagnostic.
# An ad-hoc signature ("-") is sufficient for local use.
if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "== re-signing (install_name_tool invalidates signatures) =="
    for f in "$STAGE/openfhe/lib"/*.dylib "$STAGE/openfhe/$(basename "$MODULE")"; do
        codesign --force --sign - "$f" 2>/dev/null \
            || echo "  WARNING: could not sign $(basename "$f")" >&2
    done
    codesign --verify "$STAGE/openfhe/$(basename "$MODULE")" 2>/dev/null \
        && echo "  signatures valid" \
        || echo "  WARNING: signature verification failed" >&2
    echo
fi

REMAINING="$(otool -L "$STAGE/openfhe/$(basename "$MODULE")" "$STAGE/openfhe/lib"/*.dylib \
             | grep -E '^\s' | grep -E "R\.framework|$OPENFHE_PREFIX|$SCRATCH" || true)"
if [[ -n "$REMAINING" ]]; then
    echo "  WARNING: absolute paths survive rewriting:" >&2
    echo "$REMAINING" >&2
else
    echo "  no R.framework or build-tree paths remain"
fi
echo

# -- Metadata + zip ---------------------------------------------------------
echo "== packing wheel =="
# The version lines look like `set(OPENFHE_PYTHON_VERSION_MAJOR 1)`, so
# the digits are followed by a paren -- an anchored [0-9]+$ matches
# nothing and, under pipefail, takes the whole script down silently.
VERSION="$(grep -E 'set\(OPENFHE_PYTHON_VERSION_(MAJOR|MINOR|PATCH|TWEAK)' \
           "$SRC/CMakeLists.txt" \
           | sed -E 's/.*[[:space:]]+([0-9]+)\).*/\1/' | paste -sd. -)"
if [[ ! "$VERSION" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
    echo "ERROR: could not parse version from CMakeLists (got '$VERSION')" >&2
    exit 1
fi
PYTAG="cp$(echo "$PYVER" | tr -d '.')"
ARCH="$(uname -m)"
MACOS_VER="$("$VENV_PY" -c 'import platform; v=platform.mac_ver()[0].split("."); print(f"{v[0]}_0")')"
PLATTAG="macosx_${MACOS_VER}_${ARCH}"
WHEELNAME="openfhe-${VERSION}-${PYTAG}-${PYTAG}-${PLATTAG}.whl"

DISTINFO="$STAGE/openfhe-${VERSION}.dist-info"
mkdir -p "$DISTINFO"
cat > "$DISTINFO/METADATA" <<EOF
Metadata-Version: 2.1
Name: openfhe
Version: ${VERSION}
Summary: Python bindings for OpenFHE (locally built)
Requires-Python: >=${PYVER}
Description-Content-Type: text/plain

Built from $SRC against the local OpenFHE at $OPENFHE_PREFIX by
homomorphepy/scripts/build-openfhe-wheel.sh.

Unlike the PyPI wheels, this one carries a truthful platform tag and
vendors its shared libraries with @loader_path install names, so it
does not depend on the R installation whose libomp the OpenFHE build
links.
EOF
cat > "$DISTINFO/WHEEL" <<EOF
Wheel-Version: 1.0
Generator: build-openfhe-wheel.sh
Root-Is-Purelib: false
Tag: ${PYTAG}-${PYTAG}-${PLATTAG}
EOF
echo "openfhe" > "$DISTINFO/top_level.txt"

( cd "$STAGE" && "$VENV_PY" - "$DISTINFO" <<'PY'
import base64, csv, hashlib, sys
from pathlib import Path
distinfo = Path(sys.argv[1]).name
root = Path(".")
rows = []
for p in sorted(root.rglob("*")):
    if p.is_dir() or p.name == "RECORD":
        continue
    data = p.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    rows.append([str(p), f"sha256={digest}", len(data)])
rows.append([f"{distinfo}/RECORD", "", ""])
with open(Path(distinfo) / "RECORD", "w", newline="") as fh:
    csv.writer(fh).writerows(rows)
PY
)

mkdir -p "$DIST"
rm -f "$DIST/$WHEELNAME"
( cd "$STAGE" && zip -qr "$DIST/$WHEELNAME" . )

echo "  $DIST/$WHEELNAME"
echo "  $(du -h "$DIST/$WHEELNAME" | cut -f1)"
echo
echo "== install with =="
echo "  uv pip install --python .venv/bin/python '$DIST/$WHEELNAME'"
