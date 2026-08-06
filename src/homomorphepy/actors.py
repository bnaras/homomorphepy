"""Site and Master actors for multi-site protocols.

A :class:`Site`
holds local data and a ``local_fn(data, theta)`` computing a site-level
summary; a master owns the keys and runs the protocol. The protocol
body is backend-agnostic: it calls :meth:`Master.encrypt` and
:meth:`Master.decrypt`, which the concrete masters implement.

**Secret shares live at the sites.** A threshold master that held
every ``sk_i`` itself would be a single-process simulation: convenient,
but it could not be split across processes without shipping private
keys over the wire, which would destroy the point. Here each
:class:`ThresholdSite` holds its own share and returns a *partial
decryption*; the master fuses partials and never sees a share.

**Master/worker fan-in, not a round robin.** The supported topology is
a star with the master at the center and one independent worker per
site -- what distcomp- and DataSHIELD-style deployments actually use.
There is no inter-site communication.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from homomorphepy._backend import backend
from homomorphepy.ciphertext import Ct
from homomorphepy.codec import packed_codec
from homomorphepy.context import Context

__all__ = [
    "Site",
    "Master",
    "CKKSMaster",
    "ThresholdMaster",
    "ThresholdSite",
    "make_site",
    "make_worker",
    "make_ckks_master",
    "make_threshold_master",
]

LocalFn = Callable[[Any, Any], Any]


class Site:
    """One participant: local data plus the summary it can compute.

    ``local_fn(data, theta)`` returns the site-level summary at
    ``theta``. It may return ``None`` or NaN to signal a non-evaluable
    parameter — an extreme ``theta`` that breaks the local solver — and
    the master propagates that to the optimizer.
    """

    def __init__(self, name: str, data: Any, local_fn: LocalFn):
        self.name = name
        self.data = data
        self.local_fn = local_fn
        self.public_key: Any = None

    def summary(self, theta: Any) -> Any:
        """The local summary at ``theta``, or None if non-evaluable."""
        value = self.local_fn(self.data, theta)
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    def __repr__(self) -> str:
        return f"<Site {self.name}>"


class ThresholdSite(Site):
    """A :class:`Site` that also holds a threshold secret share.

    The share never leaves the site. The master asks for a *partial
    decryption* and fuses the partials it collects.
    """

    def __init__(self, name: str, data: Any, local_fn: LocalFn):
        super().__init__(name, data, local_fn)
        self.secret_share: Any = None
        self._is_lead: bool = False

    def partial_decrypt(self, ctx: Context, ct: Any) -> Any:
        """This site's partial decryption of ``ct``.

        The lead site uses ``MultipartyDecryptLead`` and the rest use
        ``MultipartyDecryptMain``; fusion requires the lead's partial
        to come first, which :meth:`ThresholdMaster.decrypt` enforces.

        Note the Python binding shape: only the *vector* overloads are
        bound, and the arguments are (ciphertexts, key) -- reversed
        relative to OpenFHE's ``MultipartyDecryptLead``. The
        list wrapping and ``[0]`` indexing here absorb that.
        """
        if self.secret_share is None:
            raise RuntimeError(f"site {self.name!r} has no secret share")
        raw = ct.raw if isinstance(ct, Ct) else ct
        fn = ctx.MultipartyDecryptLead if self._is_lead else ctx.MultipartyDecryptMain
        return fn([raw], self.secret_share)[0]

    def __repr__(self) -> str:
        role = "lead" if self._is_lead else "main"
        return f"<ThresholdSite {self.name} ({role})>"


class Master:
    """Abstract master: owns keys, drives the protocol.

    Concrete subclasses implement :meth:`encrypt` and :meth:`decrypt`.
    ``decrypt`` takes ``length`` on every backend --
    Paillier master omitted it and S7 tolerated the arity difference
    via ``...``, but Paillier is out of scope here so the signature is
    uniform from the start (D5).
    """

    def __init__(self, name: str, ctx: Context):
        if not isinstance(ctx, Context):
            raise TypeError(
                "Master needs a homomorphepy Context (which records its "
                "scheme); build one with homomorphepy.fhe_context()"
            )
        self.name = name
        self.ctx = ctx
        self.workers: list[Site] = []

    @property
    def codec(self):
        return packed_codec(self.ctx)

    @property
    def public_key(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def encrypt(self, value: Any) -> Ct:  # pragma: no cover - overridden
        raise NotImplementedError

    def decrypt(self, ct: Any, length: int = 1) -> Any:  # pragma: no cover
        raise NotImplementedError

    # -- topology -----------------------------------------------------

    def set_workers(self, workers: Sequence[Site]) -> Master:
        """Wire workers to this master and broadcast the public key."""
        if not workers:
            raise ValueError("need at least one worker")
        self.workers = list(workers)
        for w in self.workers:
            w.public_key = self.public_key
        return self

    # -- protocol -----------------------------------------------------

    def aggregate(self, theta: Any, length: int = 1) -> Any:
        """One round of the master/worker protocol.

        Each worker computes its local summary at ``theta``; the
        summaries are encrypted, summed homomorphically, and the total
        decrypted. Returns NaN if any worker reports non-evaluable,
        returning
        ``NA_real_``.

        NaN rather than None is deliberate: scipy's optimizers raise
        ``TypeError`` on a None objective, whereas NaN at least
        propagates. Callers driving an optimizer should still wrap this
        with an explicit penalty -- see the note in D6 and the
        finite-difference caveat in the plan.
        """
        if not self.workers:
            raise RuntimeError("master has no workers; call set_workers() first")

        encrypted = []
        for w in self.workers:
            value = w.summary(theta)
            if value is None:
                return math.nan
            encrypted.append(self.encrypt(value))

        total = encrypted[0]
        for ct in encrypted[1:]:
            total = total + ct
        return self.decrypt(total, length)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


class CKKSMaster(Master):
    """Single-decrypter master: one keypair, held by the master.

    Simpler than the threshold master and adequate when the master is
    trusted to decrypt. The cryptographic guarantee strengthens with
    :class:`ThresholdMaster`, where no single party holds the key.
    """

    def __init__(self, name: str, ctx: Context, keypair: Any):
        super().__init__(name, ctx)
        self.keypair = keypair

    @property
    def public_key(self) -> Any:
        return self.keypair.publicKey

    def encrypt(self, value: Any) -> Ct:
        pt = self.codec.encode(value)
        return Ct(self.ctx.Encrypt(self.keypair.publicKey, pt), self.ctx.cc)

    def decrypt(self, ct: Any, length: int = 1) -> Any:
        raw = ct.raw if isinstance(ct, Ct) else ct
        pt = self.ctx.Decrypt(raw, self.keypair.secretKey)
        vals = self.codec.decode(pt, length)
        return vals[0] if length == 1 else vals


class ThresholdMaster(Master):
    """n-of-n threshold master; the sites hold the shares.

    The joint public key is built by chaining ``MultipartyKeyGen``
    across sites: the first generates ``(pk_1, sk_1)``, and each
    subsequent site ``i`` derives ``(pk_{1..i}, sk_i)`` from
    ``pk_{1..i-1}``. Encryption is under the final joint key.
    Decryption needs every site's partial, which the master fuses.

    This master never holds a share --
    see the module docstring.
    """

    def __init__(
        self,
        name: str,
        ctx: Context,
        joint_public_key: Any,
        sites: Sequence[ThresholdSite],
    ):
        super().__init__(name, ctx)
        self.joint_public_key = joint_public_key
        self.workers = list(sites)

    @property
    def public_key(self) -> Any:
        return self.joint_public_key

    @property
    def sites(self) -> list[ThresholdSite]:
        return [w for w in self.workers if isinstance(w, ThresholdSite)]

    def encrypt(self, value: Any) -> Ct:
        pt = self.codec.encode(value)
        return Ct(self.ctx.Encrypt(self.joint_public_key, pt), self.ctx.cc)

    def decrypt(self, ct: Any, length: int = 1) -> Any:
        """Collect partial decryptions from every site and fuse them.

        In a deployment the partials travel over the network; here they
        are ordinary objects. The lead's partial must come first --
        OpenFHE requires it, and permuting them yields garbage rather
        than an error, so the ordering is asserted.
        """
        sites = self.sites
        if not sites:
            raise RuntimeError("threshold master has no sites")
        if not sites[0]._is_lead:
            raise RuntimeError(
                "the first site must be the lead; fusion requires the "
                "lead's partial decryption first"
            )
        if any(s._is_lead for s in sites[1:]):
            raise RuntimeError("exactly one site may be the lead")

        partials = [s.partial_decrypt(self.ctx, ct) for s in sites]
        pt = self.ctx.MultipartyDecryptFusion(partials)
        vals = self.codec.decode(pt, length)
        return vals[0] if length == 1 else vals


# ---- constructors ---------------------------------------------------------


def make_site(name: str, data: Any, local_fn: LocalFn) -> Site:
    """Construct a :class:`Site`."""
    return Site(name, data, local_fn)


def make_worker(name: str, data: Any, local_fn: LocalFn) -> Site:
    """Alias of :func:`make_site`, for master/worker phrasing."""
    return make_site(name, data, local_fn)


def make_ckks_master(name: str, ctx: Context, keypair: Any) -> CKKSMaster:
    """Construct a single-decrypter :class:`CKKSMaster`."""
    return CKKSMaster(name, ctx, keypair)


def make_threshold_master(
    name: str,
    ctx: Context,
    sites: Sequence[ThresholdSite],
) -> ThresholdMaster:
    """Run the chained key generation and wire up a threshold protocol.

    Each site receives its own share; the master gets only the joint
    public key. Requires at least two sites -- one is a degenerate case
    needing no threshold scheme.

    ``MULTIPARTY`` is enabled here rather than demanded of the caller:
    a threshold master categorically needs it, so forgetting would be a
    pure footgun whose only symptom is an OpenFHE C++ error several
    calls later. ``Enable`` is idempotent, so passing
    ``features=[PKESchemeFeature.MULTIPARTY]`` to ``fhe_context()``
    as well remains correct.
    """
    sites = list(sites)
    if len(sites) < 2:
        raise ValueError("threshold key generation requires at least two sites")
    if not all(isinstance(s, ThresholdSite) for s in sites):
        raise TypeError("threshold protocols need ThresholdSite instances")

    ctx.Enable(backend().PKESchemeFeature.MULTIPARTY)

    kp = ctx.KeyGen()
    sites[0].secret_share = kp.secretKey
    sites[0]._is_lead = True
    joint_pk = kp.publicKey

    for site in sites[1:]:
        kp = ctx.MultipartyKeyGen(joint_pk)
        site.secret_share = kp.secretKey
        site._is_lead = False
        joint_pk = kp.publicKey

    master = ThresholdMaster(name, ctx, joint_pk, sites)
    for s in sites:
        s.public_key = joint_pk
    return master
