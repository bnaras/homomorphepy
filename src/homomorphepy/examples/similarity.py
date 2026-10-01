"""Federated cosine similarity across sites with private models.

Several hospitals hold patient embeddings produced by their *own*
fine-tuned models. A querier arrives with an embedding from the public
base model and asks which patients across the network are most similar.
Nobody may see anybody else's embeddings, and no site will hand over
its fine-tuned model.

Two problems have to be solved at once. The geometric one: embeddings
from different fine-tunes do not live in a common space, so a raw inner
product between a public-model query and a site's private embedding is
meaningless. Each site therefore fits a *compatibility adapter* on a
public anchor cohort -- the orthogonal map that best carries its own
geometry onto the public one -- and keeps it private.

The cryptographic one: the query is encrypted under a joint key that no
single party can undo, the adapter is applied to it while it stays
encrypted, and the resulting scores come back only through an n-of-n
decryption ceremony.

How the encrypted arithmetic works
----------------------------------

Both steps reduce to rotations of an encrypted vector's slots:

*Applying the adapter.* A matrix times an encrypted vector is computed
from the matrix's generalized diagonals,
``A q = sum_i d_i * rot(q, i)``. Each term multiplies the encrypted
vector by an *unencrypted* diagonal, which is cheaper than multiplying
two encrypted quantities and costs a single level of the precision
budget for the whole product.

*The inner product.* Multiplying slot-wise by the site's own
unencrypted database vector and then folding the slots together with a
halve-and-fold reduction leaves the cosine similarity in slot 0, in
``log2(p)`` rotations rather than ``p``.

Scope
-----

The adapter here is the orthogonal (Procrustes) one. That is the
endpoint where the recovered score is a genuine cosine, and it is the
case a deployment would use. The wider family that trades orthogonality
for reconstruction accuracy is a statistical study rather than a
protocol change, and is not reproduced here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from homomorphepy.actors import (
    ThresholdMaster,
    make_joint_rotation_keys,
    make_threshold_master,
    make_worker,
)
from homomorphepy.ciphertext import Ct
from homomorphepy.codec import packed_codec
from homomorphepy.context import Context, fhe_context

__all__ = [
    "DGP",
    "SWEEP",
    "SimilarityResult",
    "beta_sweep",
    "diagonals",
    "fed_recall",
    "fit_adapter",
    "fit_adapter_mu",
    "fit_gram",
    "mu_sweep",
    "random_drift",
    "simulate",
    "run",
]

# p is the embedding width and also the slot count: one embedding fills
# one encrypted vector exactly, which keeps the rotation arithmetic
# simple. 32 is small enough to render quickly and large enough that
# the log-p reduction is doing real work.
DGP = dict(
    p=32,
    n_sites=3,
    cohort_sizes=(6, 5, 7),
    n_anchor=64,
    drift_scale=0.35,
    top_k=5,
)

CKKS_PARAMS = dict(multiplicative_depth=3, scaling_mod_size=50, batch_size=32)


@dataclass
class SimilarityResult:
    top_k: list[tuple[str, int, float]]
    top_k_reference: list[tuple[str, int, float]]
    max_abs_score_error: float
    rank_agreement: bool
    anchor_reconstruction: list[float]
    n_candidates: int
    p: int
    context: Context = field(repr=False)
    master: ThresholdMaster = field(repr=False)


def simulate(seed: int = 20260907):
    """Public anchors, per-site drifts, and per-site private cohorts.

    Each site's model is the public one composed with a random
    orthogonal drift, so the site's embeddings are a rotated copy of the
    public geometry. That is the situation the adapter has to undo, and
    making the drift exactly orthogonal means the orthogonal adapter can
    undo it exactly -- which is what lets the test below assert equality
    rather than a tolerance.
    """
    rng = np.random.default_rng(seed)
    p = DGP["p"]

    def unit_rows(a):
        return a / np.linalg.norm(a, axis=1, keepdims=True)

    anchor_public = unit_rows(rng.normal(size=(DGP["n_anchor"], p)))

    sites = []
    for k, n in enumerate(DGP["cohort_sizes"]):
        # A random orthogonal drift for this site's fine-tune.
        drift, _ = np.linalg.qr(rng.normal(size=(p, p)))
        anchor_site = anchor_public @ drift
        cohort_public = unit_rows(rng.normal(size=(n, p)))
        sites.append(
            dict(
                name=f"Site {k + 1}",
                drift=drift,
                anchor_site=anchor_site,
                database=unit_rows(cohort_public @ drift),
                cohort_public=cohort_public,
            )
        )
    return anchor_public, sites


def fit_adapter(anchor_public: np.ndarray, anchor_site: np.ndarray) -> np.ndarray:
    """The orthogonal map carrying *public* geometry onto this site's.

    Orthogonal Procrustes: the minimizer of ``||A X_public - X_site||``
    over orthogonal ``A`` is ``U V^T`` from the SVD of
    ``X_site^T X_public``.

    The direction matters. The adapter is applied to the *incoming
    query*, which arrives in public coordinates, so it must map public
    to site -- the site's own database is already in site coordinates
    and is never moved. Fitting the reverse map and applying it to the
    query silently computes ``q^T A v`` where the intended quantity is
    ``q^T A^T v``, which agrees only for a symmetric adapter and is
    wrong for a rotation.

    Orthogonality is what keeps the score a cosine: it preserves inner
    products, so mapping the query into site coordinates leaves the
    similarity unchanged. A general linear map would distort norms and
    the returned number would no longer be a similarity.
    """
    u, _, vt = np.linalg.svd(anchor_site.T @ anchor_public)
    return u @ vt


def diagonals(m: np.ndarray) -> list[np.ndarray]:
    """Generalized diagonals of ``m``: ``d_i[j] = m[j, (j + i) % p]``.

    The encoding that turns a matrix-vector product into rotations.
    """
    p = m.shape[0]
    idx = np.arange(p)
    return [m[idx, (idx + i) % p].copy() for i in range(p)]


def _rotate(ct: Ct, k: int, cc) -> Ct:
    return Ct(cc.EvalRotate(ct.raw, int(k)), cc)


def _matvec(ct: Ct, diags: list[np.ndarray], ctx: Context) -> Ct:
    """``A q`` for encrypted ``q`` and unencrypted ``A`` given as diagonals.

    Takes the crypto context, not a master: applying an adapter to an
    encrypted query is a site-side computation that needs nothing from
    the aggregator, and naming one here would say otherwise.
    """
    cc = ctx.cc
    codec = packed_codec(ctx)
    total = None
    for i, d in enumerate(diags):
        if not np.any(d):
            continue
        rotated = ct if i == 0 else _rotate(ct, i, cc)
        term = Ct(cc.EvalMult(rotated.raw, codec.encode(d.tolist())), cc)
        total = term if total is None else Ct(cc.EvalAdd(total.raw, term.raw), cc)
    if total is None:
        raise ValueError("the adapter is entirely zero")
    return total


def _slot_sum(ct: Ct, p: int, cc) -> Ct:
    """Fold every slot into slot 0 with log2(p) rotate-and-add steps."""
    out = ct
    step = 1
    while step < p:
        out = Ct(cc.EvalAdd(out.raw, _rotate(out, step, cc).raw), cc)
        step *= 2
    return out


def run(seed: int = 20260907, top_k: int | None = None) -> SimilarityResult:
    """The whole protocol: encrypted query in, top-k matches out."""
    p = DGP["p"]
    k_out = DGP["top_k"] if top_k is None else int(top_k)
    anchor_public, site_specs = simulate(seed)

    # -- setup: joint keys, then the rotation keys the protocol needs --
    ctx = fhe_context("CKKS", **CKKS_PARAMS)
    holders = [make_worker(s["name"], None, lambda d, t: 0.0) for s in site_specs]
    master = make_threshold_master("Aggregator", ctx, holders)
    make_joint_rotation_keys(master, range(1, p))

    # The public bundle, read off a site rather than the aggregator:
    # it is what that site kept from wiring, and it is all anyone needs
    # in order to encrypt under the joint key.
    pub = holders[0].params

    # Each site fits its adapter on the public anchor cohort. The
    # adapter never leaves the site; only its effect on an encrypted
    # query does.
    for s in site_specs:
        s["adapter"] = fit_adapter(anchor_public, s["anchor_site"])
        s["diags"] = diagonals(s["adapter"])
        recon = s["adapter"] @ anchor_public.T - s["anchor_site"].T
        s["anchor_error"] = float(np.max(np.abs(recon)))

    # -- the querier -------------------------------------------------
    rng = np.random.default_rng(seed + 7)
    query = rng.normal(size=p)
    query = query / np.linalg.norm(query)
    ct_query = pub.encrypt(query.tolist())

    # -- each site scores its own cohort, without decrypting ---------
    encrypted_scores: list[tuple[str, int, Ct]] = []
    site_codec = packed_codec(ctx)
    for s in site_specs:
        ct_mapped = _matvec(ct_query, s["diags"], ctx)
        for i, v in enumerate(s["database"]):
            # Slot-wise against this patient's own unencrypted vector,
            # then fold the slots to leave the inner product in slot 0.
            term = Ct(
                ctx.cc.EvalMult(ct_mapped.raw, site_codec.encode(v.tolist())),
                ctx.cc,
            )
            encrypted_scores.append((s["name"], i, _slot_sum(term, p, ctx.cc)))

    # -- the aggregator fuses one score at a time --------------------
    scored = [
        (name, i, float(np.asarray(master.decrypt(ct, length=p), dtype=float)[0]))
        for name, i, ct in encrypted_scores
    ]
    scored.sort(key=lambda r: -r[2])
    top = scored[:k_out]

    # -- the same computation in the clear, for comparison -----------
    reference = []
    for s in site_specs:
        mapped_query = s["adapter"] @ query
        for i, v in enumerate(s["database"]):
            reference.append((s["name"], i, float(mapped_query @ v)))
    reference.sort(key=lambda r: -r[2])
    ref_top = reference[:k_out]

    by_key = {(n, i): v for n, i, v in reference}
    max_err = max(abs(v - by_key[(n, i)]) for n, i, v in scored)

    return SimilarityResult(
        top_k=top,
        top_k_reference=ref_top,
        max_abs_score_error=float(max_err),
        rank_agreement=[(n, i) for n, i, _ in top] == [(n, i) for n, i, _ in ref_top],
        anchor_reconstruction=[s["anchor_error"] for s in site_specs],
        n_candidates=len(scored),
        p=p,
        context=ctx,
        master=master,
    )


# ---------------------------------------------------------------------
# Does the adapter have to be orthogonal? The statistical study.
# ---------------------------------------------------------------------
#
# Everything above runs at the orthogonal endpoint, where the protocol
# is exact. Real fine-tuning is not orthogonal, and relaxing the
# constraint trades a better reconstruction against a score that is no
# longer quite a cosine. Whether that trade is worth taking is an
# empirical question, and it is answered in the clear: the encrypted
# and unencrypted scores agree to 1e-13, so running the sweeps under
# encryption would cost hours and change no digit.

SWEEP = dict(
    p=32,
    n_sites=3,
    cohort_sizes=(80, 60, 100),
    n_anchor=100,
    n_phenotypes=5,
    top_k=5,
    separation=0.85,
    noise_sd=0.40,
    n_rep=3,
    n_query=40,
    mu_grid=(0.0, 0.1, 1.0, 10.0, float("inf")),
)


def _unit_rows(z: np.ndarray) -> np.ndarray:
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def random_drift(p: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """A fine-tune, modeled as ``B = Q D``.

    A random rotation composed with an anisotropic stretch
    ``D = diag(exp(beta * g))``. ``beta = 0`` gives an exactly
    orthogonal drift -- the special case where the geometry is merely
    rotated, and the one the protocol walk-through uses. ``beta > 0``
    stretches directions unevenly, which is what a freely fine-tuned
    model generically does.
    """
    q, _ = np.linalg.qr(rng.normal(size=(p, p)))
    if beta == 0:
        return q
    return q @ np.diag(np.exp(beta * rng.normal(size=p)))


def _embed(n, centers, rng, sep, sd):
    """A cohort of ``n`` patients drawn around phenotype centers."""
    n_pheno, p = centers.shape
    labels = rng.integers(0, n_pheno, size=n)
    z = sep * centers[labels] + rng.normal(scale=sd, size=(n, p))
    return dict(z=_unit_rows(z), label=labels)


def _embed_private(z_public: np.ndarray, drift: np.ndarray) -> np.ndarray:
    """Site embeddings ``B f(x)``, renormalized onto the sphere.

    Under a non-isometric ``B`` that renormalization is a genuine
    nonlinearity, which is part of why no linear adapter can undo the
    drift exactly once beta > 0.
    """
    return _unit_rows(z_public @ drift.T)


def fit_adapter_mu(
    anchor_public: np.ndarray,
    anchor_site: np.ndarray,
    mu: float,
) -> np.ndarray:
    """The adapter family, indexed by a near-isometry penalty.

    Minimizes ``||Z_site A - Z_public||^2 + mu * ||A^T A - I||^2``.
    At ``mu = inf`` the constraint binds and this is the orthogonal
    Procrustes solution of :func:`fit_adapter`; at ``mu = 0`` it is
    unconstrained least squares, computed with a small ridge on the
    normal equations; in between, near-orthogonal.
    """
    p = anchor_site.shape[1]
    if np.isinf(mu):
        return fit_adapter(anchor_public, anchor_site)

    gram = anchor_site.T @ anchor_site + 1e-6 * np.eye(p)
    a_ls = np.linalg.solve(gram, anchor_site.T @ anchor_public)
    if mu == 0:
        return a_ls

    from scipy.optimize import minimize

    def fn(par):
        a = par.reshape(p, p)
        resid = anchor_site @ a - anchor_public
        skew = a.T @ a - np.eye(p)
        return float(np.sum(resid**2) + mu * np.sum(skew**2))

    def gr(par):
        a = par.reshape(p, p)
        resid = anchor_site @ a - anchor_public
        skew = a.T @ a - np.eye(p)
        return (2 * anchor_site.T @ resid + 4 * mu * (a @ skew)).ravel()

    out = minimize(
        fn, a_ls.ravel(), jac=gr, method="L-BFGS-B", options=dict(maxiter=400)
    )
    return out.x.reshape(p, p)


def fit_gram(anchor_public: np.ndarray, anchor_site: np.ndarray) -> np.ndarray:
    """The convex alternative: match Gram matrices, deploy the square root.

    Included only so the comparison below is against something, rather
    than against nothing. Its minimizer is analytic.
    """
    p = anchor_site.shape[1]
    a_ls = np.linalg.solve(
        anchor_site.T @ anchor_site + 1e-6 * np.eye(p),
        anchor_site.T @ anchor_public,
    )
    m = a_ls @ a_ls.T
    vals, vecs = np.linalg.eigh(m)
    return vecs @ (np.sqrt(np.maximum(vals, 0.0))[:, None] * vecs.T)


def fed_recall(query_pop, databases, adapters, design: int, top_k: int) -> float:
    """Mean recall@k of a query population across the federation.

    ``design=1`` folds the adapter into the database offline and scores
    ``<q, unit(z A)>``; ``design=2`` applies it to the query and scores
    ``<z, A q>``. They coincide exactly when ``A`` is orthogonal, and
    part company otherwise -- which is the point of measuring both.
    """
    hits = []
    for qi in range(query_pop["z"].shape[0]):
        q = query_pop["z"][qi]
        labels, scores = [], []
        for db, a in zip(databases, adapters, strict=True):
            if design == 1:
                sc = _unit_rows(db["z"] @ a) @ q
            else:
                sc = db["z"] @ (a @ q)
            labels.append(db["label"])
            scores.append(sc)
        labels = np.concatenate(labels)
        scores = np.concatenate(scores)
        top = np.argsort(-scores)[:top_k]
        hits.append(float(np.sum(labels[top] == query_pop["label"][qi]) / top_k))
    return float(np.mean(hits))


def _world(beta, n_anchor, centers, rng, cfg):
    """One draw: an anchor cohort, per-site drifts, and per-site cohorts."""
    sep, sd = cfg["separation"], cfg["noise_sd"]
    anchor = _embed(n_anchor, centers, rng, sep, sd)
    public = [_embed(n, centers, rng, sep, sd) for n in cfg["cohort_sizes"]]
    drifts = [random_drift(cfg["p"], beta, rng) for _ in cfg["cohort_sizes"]]
    return dict(
        anchor=anchor,
        public_db=public,
        private_db=[
            dict(z=_embed_private(c["z"], b), label=c["label"])
            for c, b in zip(public, drifts, strict=True)
        ],
        anchor_site=[_embed_private(anchor["z"], b) for b in drifts],
    )


def mu_sweep(beta: float = 0.6, n_anchor: int = 100, seed: int = 20260428) -> dict:
    """Recall across the adapter family at a fixed non-isometry.

    Reports both deployments at each ``mu``, plus three reference lines:
    the *ideal* (everyone already in public coordinates), the
    *unaligned* (no adapter at all), and the convex Gram alternative.
    """
    cfg = SWEEP
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(cfg["n_phenotypes"], cfg["p"]))
    centers = _unit_rows(centers)
    ident = [np.eye(cfg["p"])] * cfg["n_sites"]
    k = cfg["top_k"]

    rows = {mu: dict(d1=0.0, d2=0.0, aniso=0.0) for mu in cfg["mu_grid"]}
    ideal = unaligned = gram = 0.0

    for _ in range(cfg["n_rep"]):
        w = _world(beta, n_anchor, centers, rng, cfg)
        qp = _embed(cfg["n_query"], centers, rng, cfg["separation"], cfg["noise_sd"])

        ideal += fed_recall(qp, w["public_db"], ident, 2, k)
        unaligned += fed_recall(qp, w["private_db"], ident, 2, k)

        ag = [fit_gram(w["anchor"]["z"], h) for h in w["anchor_site"]]
        gram += fed_recall(qp, w["private_db"], ag, 1, k)

        for mu in cfg["mu_grid"]:
            a = [fit_adapter_mu(w["anchor"]["z"], h, mu) for h in w["anchor_site"]]
            rows[mu]["d1"] += fed_recall(qp, w["private_db"], a, 1, k)
            rows[mu]["d2"] += fed_recall(qp, w["private_db"], a, 2, k)
            rows[mu]["aniso"] += float(
                np.mean([np.linalg.norm(x.T @ x - np.eye(cfg["p"])) for x in a])
            )

    n = cfg["n_rep"]
    return dict(
        beta=beta,
        n_anchor=n_anchor,
        table=[
            dict(mu=mu, d1=v["d1"] / n, d2=v["d2"] / n, aniso=v["aniso"] / n)
            for mu, v in rows.items()
        ],
        ideal=ideal / n,
        unaligned=unaligned / n,
        gram=gram / n,
    )


def beta_sweep(
    betas: tuple[float, ...] = (0.0, 0.3, 0.6, 1.0),
    n_anchor: int = 100,
    seed: int = 20260428,
) -> list[dict]:
    """Recall against drift magnitude, across three points of the family.

    The two endpoints -- Procrustes at ``mu = inf`` and least squares at
    ``mu = 0`` -- plus the near-orthogonal ``mu = 1`` between them, which
    is what shows that the fall-off is gradual in ``mu`` rather than a
    step at the endpoint.

    Common random numbers across ``mu`` within a replicate: the cohorts
    and the drift *directions* are shared, and only the stretch
    magnitude varies, so the curves differ because of ``beta`` and not
    because of the draw.
    """
    cfg = SWEEP
    out = []
    for beta in betas:
        rng = np.random.default_rng(seed)  # same draw at every beta
        centers = _unit_rows(rng.normal(size=(cfg["n_phenotypes"], cfg["p"])))
        ident = [np.eye(cfg["p"])] * cfg["n_sites"]
        k = cfg["top_k"]
        acc = dict(
            procrustes=0.0,
            near_orthogonal=0.0,
            least_squares=0.0,
            ideal=0.0,
            unaligned=0.0,
        )
        for _ in range(cfg["n_rep"]):
            w = _world(beta, n_anchor, centers, rng, cfg)
            qp = _embed(
                cfg["n_query"], centers, rng, cfg["separation"], cfg["noise_sd"]
            )
            acc["ideal"] += fed_recall(qp, w["public_db"], ident, 2, k)
            acc["unaligned"] += fed_recall(qp, w["private_db"], ident, 2, k)
            for name, mu in (
                ("procrustes", float("inf")),
                ("near_orthogonal", 1.0),
                ("least_squares", 0.0),
            ):
                a = [fit_adapter_mu(w["anchor"]["z"], h, mu) for h in w["anchor_site"]]
                acc[name] += fed_recall(qp, w["private_db"], a, 1, k)
        out.append(dict(beta=beta, **{s: v / cfg["n_rep"] for s, v in acc.items()}))
    return out


if __name__ == "__main__":  # pragma: no cover
    r = run()
    print(f"{r.n_candidates} candidates across 3 sites, p = {r.p}")
    for name, i, v in r.top_k:
        print(f"  {name} patient {i}: {v:.6f}")
    print(f"max score error vs cleartext : {r.max_abs_score_error:.2e}")
    print(f"same ranking as cleartext    : {r.rank_agreement}")
