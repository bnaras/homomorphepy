# homomorphepy

Multi-site privacy-preserving statistics over homomorphic encryption —
the Python twin of the R package
[homomorpheR](https://github.com/bnaras/homomorpheR), built on
[openfhe-python](https://github.com/openfheorg/openfhe-python).

Both packages sit on the same OpenFHE C++ library and run the same
worked examples: threshold-FHE Cox regression, maximum likelihood
across sites, consensus ADMM, secure inference, aggregation, similarity
search, and exact integer query counting under BFV.

**Status: early development.** The fixture layer and the backend facade
are in place; the protocol actors and the worked examples are not yet.

## Why the examples share fixtures with R

R's `rpois`, `rbinom`, `sample` and `rnorm` algorithms have no numpy
equivalent — no amount of seeding makes Python reproduce R's stream.
Eight of the twelve worked examples simulate their inputs that way, so
re-simulating here would compute on *different data* and leave nothing
meaningful to compare against the R results.

Instead the inputs are exported once from R and both languages read the
same bytes. Everything in `homomorphepy.fixtures` asserts rather than
infers: dtypes and category orders come from a manifest, and every file
is checked against a SHA-256 digest on load. A stale fixture fails
loudly, because the alternative — a silent mismatch — would make a
Python-vs-R disagreement look like a binding defect.

That also means the fixture layer needs no crypto backend, so the
package can be developed and tested on platforms where `openfhe`
cannot currently be installed.

```python
from homomorphepy import load_dlbcl, load_dlbcl_gex, site_order

df = load_dlbcl()  # dtypes and site order forced
gex, rows, cols = load_dlbcl_gex()  # bit-exact float64
site_order()  # ['GCB', 'ABC', 'Type III']
```

Site order is protocol semantics, not presentation: sites are visited
in that order and the first one is the lead decryptor in the threshold
decryption. Sorting the sites alphabetically would silently permute the
protocol, so the order is declared and asserted.

## Installing the crypto backend

`openfhe` is an **optional** dependency, deliberately. Its PyPI wheels
are tagged `py3-none-any` but contain
`cpython-312-x86_64-linux-gnu` binaries with Ubuntu-specific shared
libraries, so `pip install` reports success on macOS and on other
CPython versions and `import openfhe` fails afterwards. Making it a
hard dependency would leave `uv sync` green and every test red.

```
# Linux x86_64, CPython 3.12 (Ubuntu 24.04)
uv pip install 'openfhe==1.5.1.0.24.4'   # Ubuntu 22.04: ...0.22.4
```

Never pin with a wildcard: the Ubuntu release is encoded in the
*version*, not the wheel tag, so `openfhe==1.5.1.0.*` resolves to the
24.04 build on 22.04 too. Everywhere else, build openfhe-python from
source against a local OpenFHE — see `docs/install.md`.

### Threads

OpenFHE latches its thread count when the shared library loads, and its
hot regions carry explicit `num_threads` clauses that override the
OpenMP thread-count variable afterwards. A cap therefore has to be in
place *before* the first `import openfhe`:

```python
from homomorphepy import set_thread_env

set_thread_env(2)  # must run before anything imports openfhe
```

In CI, set `OMP_NUM_THREADS` in the job environment instead — more
robust than relying on import order. Upstream openfhe-python binds no
thread-control API; that is filed as a defect, and this helper will
switch to calling it once a release exposes it.

## Development

```
uv sync                  # dev dependencies
uv run pytest            # fixture + facade tests (no backend needed)
uv run ruff check .
```

Fixtures are staged from the monorepo:

```
bash ../../fixtures/sync_fixtures.sh src/homomorphepy/fixtures
```

That regenerates them from homomorpheR, verifies them, stages them, and
re-verifies the staged copy. Nothing is copied unless verification
passes.

## License

MIT
