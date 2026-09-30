# homomorphepy

Multi-site privacy-preserving statistics over homomorphic encryption,
built on [OpenFHE](https://openfhe.org) via
[openfhe-python](https://github.com/openfheorg/openfhe-python).

Several sites hold data they will not share. They are willing to
compute a *joint* result, provided no party — including whoever
coordinates the computation — learns anything about an individual
site's contribution. Each site computes a summary of its own data,
encrypts it, and sends the ciphertext; the coordinator adds the
ciphertexts without decrypting them and recovers only the total. Under
threshold keys the decryption key is split across the sites, so no
single party can decrypt anything at all.

What makes this practical for statistics is that the analysis code does
not change. An optimizer handed an objective that happens to route
through encryption converges to the same estimate it would have reached
on pooled data.

```python
from homomorphepy import fhe_context, packed_codec, Ct

cc = fhe_context("BFV", plaintext_modulus=65537, multiplicative_depth=1)
keys = cc.KeyGen()
codec = packed_codec(cc)

private_counts = [46, 15, 52]          # each known only to its own site

cts = [Ct(cc.Encrypt(keys.publicKey, codec.encode(n)), cc.cc)
       for n in private_counts]

codec.decode(cc.Decrypt(sum(cts).raw, keys.secretKey), 1)[0]   # 113
```

## Worked examples

Each is a module under `homomorphepy.examples` exposing `run()`, and a
documentation page that builds the same protocol step by step:

| Example | What it shows |
|---|---|
| `aggregation` | Encrypted counting under a single-decrypter coordinator. Exact under BFV. |
| `query_count` | The same count under threshold keys, where nobody can decrypt alone. |
| `mle` | An unmodified optimizer driving an encrypted objective. |
| `cox` | Stratified Cox regression across three sites, single-decrypter or threshold. |
| `cox_lasso` | The full pipeline on gene expression: encrypted standardization, screening, and a penalized fit by consensus ADMM. |
| `secure_inference` | A lab scoring patients it cannot see — and the attack this does *not* prevent. |
| `encrypted_regression` | The same with a logistic model: a sigmoid evaluated on encrypted values. |
| `similarity` | Retrieval across sites whose models disagree, under threshold keys and joint rotation keys. |
| `dp` | What adding differential privacy on top costs, for the Cox fit and for consensus ADMM. A demonstration, not a recommendation. |

The documentation also has a Precision page on what it means for an
encrypted result to be correct.

```
uv run python -m homomorphepy.examples.query_count
```

## Data

**Simulated examples draw their own data.** Where the setting is a
data-generating process rather than a measurement, each run simulates
afresh. The claim being made is that the encrypted protocol reproduces
the cleartext answer *on whatever data it was given*, which is stronger
than reproducing one stored dataset — and the tests check it across
several draws.

**Measured data ships with the package.** The Rosenwald DLBCL cohort
used by the Cox examples is what it is; there is no draw to repeat, so
every run reads the same bytes. The expression matrix travels as raw
float64 rather than text, because the screening step ranks 6416 probes
and keeps 100, and probes nearly tied at that boundary can swap under a
one-ulp perturbation that a text round-trip would introduce.

```python
from homomorphepy import load_dlbcl, load_dlbcl_gex, site_order

df = load_dlbcl()                    # dtypes and site order forced
gex, rows, cols = load_dlbcl_gex()   # bit-exact float64
site_order()                         # ['GCB', 'ABC', 'Type III']
```

Everything in `homomorphepy.fixtures` asserts rather than infers:
dtypes and category orders come from a manifest, and every file is
checked against a SHA-256 digest on load. Site order is protocol
semantics, not presentation — sites are visited in that order and the
first is the lead decryptor in the threshold decryption, so sorting
them alphabetically would silently permute the protocol.

That layer needs no crypto backend, so the package can be developed and
tested on platforms where `openfhe` cannot currently be installed.

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

## Documentation

The site is built with Quarto and every page executes: each number in
the prose comes from code that ran during the render, never from a
value typed by hand.

```
cd docs && OMP_NUM_THREADS=2 uv run quarto render
```

Two computations are too slow to run on every render — the Cox-lasso
consensus ADMM (~30 min) and the differential-privacy sweep (~10 min).
Those are recorded once by the scripts in `docs/_recorded/`, which
write the JSON the pages read. Nothing is hidden: re-running the script
regenerates the numbers.

## Development

```
uv sync                  # dev dependencies
uv run pytest            # add -m slow for the ADMM pipeline and DP sweeps
uv run ruff check .
```

## License

MIT
