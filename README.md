# homomorphepy

Multi-site privacy-preserving statistics over homomorphic encryption,
built on [OpenFHE](https://openfhe.org) via
[openfhe-python](https://github.com/openfheorg/openfhe-python).

Several sites hold data they will not share. They are willing to
compute a *joint* result, provided no party — including whoever
coordinates the computation — learns anything about an individual
site's contribution. Each site computes a summary of its own data,
encrypts it, and sends the encrypted value; the coordinator adds the
encrypted values without decrypting them and recovers only the total. Under
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
| `dp` | What adding site-side Gaussian noise on top costs, for the Cox fit and for consensus ADMM. A demonstration, not a privacy guarantee or a recommendation. |

The documentation also has a Precision page on what it means for an
encrypted result to be correct.

```
uv run python -m homomorphepy.examples.query_count
```

## Data

Simulated examples draw fresh data on each run. The Rosenwald DLBCL
cohort used by the Cox examples ships with the package; its expression
matrix is stored as float64 binary rather than text.

```python
from homomorphepy import load_dlbcl, load_dlbcl_gex, site_order

df = load_dlbcl()
gex, rows, cols = load_dlbcl_gex()
site_order()                         # ['GCB', 'ABC', 'Type III']
```

Column types, category orders, and site order come from a manifest,
and each file is checked against a SHA-256 digest on load. Sites are
visited in the stored order, and the first site leads the threshold
decryption.

## Installing the crypto backend

`openfhe` is an optional dependency. Its PyPI wheels are tagged
`py3-none-any` but contain CPython 3.12, x86_64 Linux binaries built
on Ubuntu, so `pip install` succeeds on other platforms and
`import openfhe` then fails.

```
# Linux x86_64, CPython 3.12 (Ubuntu 24.04)
uv pip install 'openfhe==1.5.1.0.24.4'   # Ubuntu 22.04: ...0.22.4
```

The Ubuntu release is encoded in the version, so pin all six
components; `openfhe==1.5.1.0.*` resolves to the 24.04 build on 22.04.
On other platforms, build
[openfhe-python](https://github.com/openfheorg/openfhe-python) from
source against a local OpenFHE and install the resulting wheel.

### Threads

OpenFHE fixes its thread count when the shared library loads, so a cap
has to be set before the first `import openfhe`:

```python
from homomorphepy import set_thread_env

set_thread_env(2)  # must run before anything imports openfhe
```

Setting `OMP_NUM_THREADS` in the environment before starting Python
has the same effect.

## Documentation

The documentation is at <https://bnaras.github.io/homomorphepy>. Each
page is also available there as an executed Jupyter notebook, linked
from the page under "Other Formats".

```
cd docs && OMP_NUM_THREADS=2 uv run quarto render
```

The Cox-lasso consensus ADMM and the Gaussian-noise sweep are
precomputed by the scripts in `docs/_recorded/`; the pages read their
JSON output.

## Development

```
uv sync                  # dev dependencies
uv run pytest            # add -m slow for the ADMM pipeline and DP sweeps
uv run ruff check .
```

## Citing

The package and its protocols are described in

> Narasimhan, B. (2026). Fully Homomorphic Encryption for Statistical
> Modeling. arXiv:2610.04163 [stat.CO].
> <https://arxiv.org/abs/2610.04163>

`CITATION.cff` in this repository carries the same reference in
machine-readable form.

## License

MIT
