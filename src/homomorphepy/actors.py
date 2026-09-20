"""Site and Master actors for multi-site protocols.

A :class:`Site` holds local data and a ``local_fn(data, theta)``
computing a site-level summary; a master orchestrates the protocol. The
protocol body is backend-agnostic: it reaches sites through
:meth:`Site.contribute` and recovers the total through
:meth:`Master.decrypt`, which the concrete masters implement.

**Sites are autonomous once wired.** A site is handed its
:class:`~homomorphepy.params.PublicParams` exactly once, at setup, and
from then on it computes *and encrypts* on its own. It holds no
reference to whoever aggregates its answers and needs none. There are
exactly two moments at which anything passes between a coordinating
party and a site: setup, and a round. A master has no ``encrypt``
method at all — encryption needs only public material, so it belongs to
whoever holds the bundle, and naming a party in it would be wrong.

**Secret shares live at the sites.** A threshold master that held every
``sk_i`` itself would be a single-process simulation: convenient, but it
could not be split across processes without shipping private keys over
the wire, which would destroy the point. Each site generates its own
share during :meth:`Site.keygen_round`, keeps it, and returns only a
public key; the master fuses partial decryptions and never sees a
share.

**Master/worker fan-in, not a round robin.** The supported topology is
a star with the master at the center and one independent worker per
site -- what distcomp- and DataSHIELD-style deployments actually use.
There is no inter-site communication.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from homomorphepy.ciphertext import Ct, unwrap
from homomorphepy.codec import packed_codec
from homomorphepy.context import Context, Scheme
from homomorphepy.params import OpenFHEParams, PublicParams

__all__ = [
    "Site",
    "RemoteSite",
    "Master",
    "CKKSMaster",
    "ThresholdMaster",
    "SiteUnavailable",
    "make_site",
    "make_worker",
    "make_ckks_master",
    "make_threshold_master",
    "make_joint_rotation_keys",
]

LocalFn = Callable[[Any, Any], Any]


class SiteUnavailable(RuntimeError):
    """A site could not be reached.

    Raised by a :class:`RemoteSite` implementation when a transport,
    authentication or timeout failure stops it from answering. This is
    **not** the same event as returning ``None``, which means the
    requested ``theta`` is non-evaluable at a site that answered
    perfectly well.

    Returning ``None`` tells the optimizer to back off and try a
    different parameter, which is the right response to a failed local
    solve and no response at all to an unreachable service. Raising
    this aborts the round instead, because continuing would sum over a
    different set of sites and silently change the objective between
    optimizer iterations.

    When a master re-raises this it attaches ``site_name`` and **not**
    the site object: a site drags its data, and under threshold keys
    its key share, into anything that logs or pickles the exception.
    """

    def __init__(self, message: str, site_name: str | None = None):
        super().__init__(message)
        self.site_name = site_name


def _check_name(name: Any) -> str:
    """A party's name appears in every error message about it.

    An empty, missing or non-string name makes those messages useless
    exactly when they matter, so it is rejected at construction rather
    than three rounds into a protocol.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("a site or master needs a single non-empty name")
    return name


class Site:
    """One participant: local data plus the summary it can compute.

    ``local_fn(data, theta)`` returns the site-level summary at
    ``theta``. It may return ``None`` or NaN to signal a non-evaluable
    parameter — an extreme ``theta`` that breaks the local solver — and
    the master propagates that to the optimizer.

    A site is given its public parameters once, by
    :meth:`set_public_params`, and encrypts with them in
    :meth:`contribute`. Under threshold keys it also generates and keeps
    its own secret share in :meth:`keygen_round`. The same class serves
    both protocols: what makes a site a threshold party is having run a
    key-generation round, not being of a different type.

    A ``Site`` demonstrates the protocol's roles inside one Python
    process. It is not a deployment boundary: its data, and its key
    share, are objects in this process and anything else in this
    process can reach them. Separating the parties for real means
    separately controlled processes, which is what :class:`RemoteSite`
    is for.
    """

    # A RemoteSite overrides contribute() and has no local function to
    # call, so the callable check below applies only to co-located
    # sites. A non-callable here would otherwise fail at the first
    # round rather than at construction.
    _needs_local_fn = True

    def __init__(self, name: str, data: Any, local_fn: LocalFn):
        self.name = _check_name(name)
        if self._needs_local_fn and not callable(local_fn):
            raise TypeError(f"site {name!r} needs a callable local_fn(data, theta)")
        self.data = data
        self.local_fn = local_fn
        self._params: PublicParams | None = None
        self._share: Any = None
        self._ctx: Context | None = None

    # -- setup ---------------------------------------------------------

    def set_public_params(self, params: PublicParams) -> Site:
        """Receive the public parameters this site will encrypt under.

        The setup step, and one of only two moments at which anything
        passes between a coordinating party and a site — the other
        being a round itself, which carries a query out and a
        ciphertext back.

        Called for you by :meth:`Master.set_workers` and
        :func:`make_threshold_master`. You would call it directly only
        when writing a :class:`RemoteSite` subclass.

        Receiving the *same* parameters again is harmless and allowed.
        Receiving different ones is refused: a site that silently
        switched keys would go on answering its first coordinator in a
        key that coordinator cannot read — under CKKS that surfaces as
        an approximation-error abort, and under BFV or BGV as a
        plausible wrong integer with nothing raised. Build a fresh
        site instead; they are cheap.
        """
        if not isinstance(params, PublicParams):
            raise TypeError("params must be a PublicParams object")
        held = self._params
        if held is not None and held.tag != params.tag:
            raise ValueError(
                f"site {self.name!r} already holds different public "
                "parameters. It would go on answering its first coordinator "
                "in a key that coordinator cannot read; under BFV or BGV "
                "that returns a plausible wrong integer and raises nothing. "
                "Build a fresh site with make_worker()."
            )
        self._params = params
        return self

    @property
    def params(self) -> PublicParams:
        """The public parameters this site holds.

        Encryption needs only this, so a party that has it is
        self-sufficient, and any other party that will encrypt under
        the same key — a querier that is not itself a site, say — can
        be handed a copy. Asking a site for its bundle involves no
        coordinator.

        Raises rather than returning ``None`` for a caller to encrypt
        with.
        """
        if self._params is None:
            raise RuntimeError(
                f"site {self.name!r} has no public parameters. A site is "
                "given them once, when it is wired with set_workers() or "
                "taken through make_threshold_master(). Do that first."
            )
        return self._params

    def _clear(self) -> None:
        """Undo participation in a ceremony that did not complete.

        Best effort, and deliberately local: for a :class:`RemoteSite`
        this reaches the proxy only, which is why
        :func:`make_threshold_master` tells remote implementers to
        tolerate a repeated ceremony.
        """
        self._params = None
        self._share = None
        self._ctx = None

    # -- a round -------------------------------------------------------

    def contribute(self, theta: Any) -> Ct | None:
        """This site's **encrypted** contribution at ``theta``.

        The single call the protocol runner makes on a site. The
        computation is entirely local: a site needs nothing at call
        time beyond ``theta``, its own data, and what it already holds.
        What leaves is already a ciphertext, so an individual site's
        cleartext contribution never reaches the aggregator — that is
        the property the whole protocol rests on.

        ``None`` is the one permitted plaintext reply, signalling that
        ``theta`` is non-evaluable here; CKKS has no representation for
        it, so it cannot be encrypted. A site that cannot be *reached*
        raises :class:`SiteUnavailable` instead.
        """
        value = self.local_fn(self.data, theta)
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return self.params.encrypt(value)

    # -- threshold keys ------------------------------------------------

    def keygen_round(self, ctx: Context, prev_pk: Any = None) -> Any:
        """This site's step in the threshold key-generation chain.

        The site derives its own secret share from its predecessor's
        cumulative public key, **keeps the share**, and returns only
        the new cumulative public key. The share is generated here and
        is never a return value, so no other party can hold it.

        ``prev_pk`` is ``None`` for the lead site, which starts the
        chain with a fresh keypair.
        """
        kp = ctx.KeyGen() if prev_pk is None else ctx.MultipartyKeyGen(prev_pk)
        # The share stays here. Only the public half is returned.
        self._ctx = ctx
        self._share = kp.secretKey
        return kp.publicKey

    @property
    def has_share(self) -> bool:
        """Whether this site took part in a key-generation ceremony."""
        return self._share is not None

    @property
    def secret_share(self) -> Any:
        """This site's own threshold share.

        Exposed so a demonstration can *show* that the share is here
        and not at the master. It never travels: the protocol moves
        partial decryptions, which :meth:`partial_decrypt` produces.
        """
        return self._share

    def partial_decrypt(self, ciphertext: Any, lead: bool = False) -> Any:
        """This site's partial decryption of ``ciphertext``.

        Under threshold keys no party can decrypt alone. A ciphertext
        is sent to each site; each applies **its own** share and
        returns a partial, and the partials are fused by
        :meth:`ThresholdMaster.decrypt`.

        Whether a site plays the ``lead`` role is fixed by its position
        in the key-generation chain, so it arrives with the request:
        the site does not choose and does not need to know who is
        asking.

        Note the Python binding shape: only the *vector* overloads are
        bound, and the arguments are (ciphertexts, key) -- reversed
        relative to OpenFHE's ``MultipartyDecryptLead``. The list
        wrapping and ``[0]`` indexing here absorb that.
        """
        if self._share is None:
            raise RuntimeError(
                f"site {self.name!r} holds no secret share; only sites that "
                "took part in make_threshold_master() can produce a partial "
                "decryption"
            )
        # The site checks for itself, with the joint key it was given at
        # setup, that this ciphertext belongs to the protocol it joined.
        # Applying its share to anything else is work it did not agree
        # to, and the requester is not a party it has reason to trust.
        # Nothing here is asked of anyone: the tag and the joint public
        # key are both already in hand.
        if self._params is not None:
            self._params.check_encrypted(ciphertext, "partially decrypt")

        ctx = self._ctx
        assert ctx is not None  # set together with _share in keygen_round
        raw = unwrap(ciphertext)
        fn = ctx.MultipartyDecryptLead if lead else ctx.MultipartyDecryptMain
        return fn([raw], self._share)[0]

    def __repr__(self) -> str:
        state = []
        if self._params is not None:
            state.append("wired")
        if self._share is not None:
            state.append("holds a share")
        return f"<Site {self.name}{' (' + ', '.join(state) + ')' if state else ''}>"


class RemoteSite(Site, ABC):
    """A site whose contribution is produced outside this process.

    Abstract. homomorphepy deliberately ships **no** implementation:
    transports differ too much, and a crypto package has no business
    carrying an HTTP client. Subclass it, add whatever your transport
    needs, and override :meth:`set_public_params`, :meth:`contribute`,
    and — for threshold protocols — :meth:`keygen_round` and
    :meth:`partial_decrypt`::

        class HttpSite(RemoteSite):
            def __init__(self, name, url):
                super().__init__(name, data=None, local_fn=None)
                self.url = url

            def set_public_params(self, params):
                ...  # POST the context and key; the far end stores them

            def contribute(self, theta):
                ...  # call self.url with theta; the far end encrypts

    What this class is, and is not
    ------------------------------
    A ``RemoteSite`` is an **architectural seam with a documented
    contract**, not a trust boundary the package establishes. Three
    cases are worth keeping apart:

    * A :class:`Site` demonstration. Data, key shares, sites and the
      aggregating party are all objects in one process. The classes
      model the protocol's *roles*; they create no process or trust
      boundary.
    * A single-decrypter deployment. Each site returns a ciphertext,
      but a :class:`CKKSMaster` holds the secret key and could decrypt
      an individual contribution. "Only the aggregate is decrypted"
      describes what :meth:`Master.aggregate` does, not something the
      cryptography enforces.
    * A remote threshold deployment. Separately controlled endpoints
      keep their own shares and return ciphertexts or partial
      decryptions. Here the party boundary is real — provided *your*
      transport, authentication, endpoint code and key storage
      implement it. homomorphepy supplies none of those, and detects no
      deliberately dishonest reply.

    The contract an implementation must honor
    -----------------------------------------
    * **Provision the far end at setup.** Implement
      :meth:`set_public_params` to send the context and key to the
      endpoint and have it retain them. Only public material travels.
    * **Return a ciphertext, never a plain number.** The remote end was
      given the parameters when it was wired, so it encrypts *before*
      the value crosses the wire. ``None`` is the one permitted
      plaintext reply; the aggregator consequently learns which
      ``theta`` a site could not evaluate, and that residual side
      channel is documented on :meth:`Master.aggregate`.
    * **Distinguish "non-evaluable" from "unreachable".** ``None``
      means *this theta broke my solver*. A network, authentication or
      timeout failure is a different event: raise
      :class:`SiteUnavailable`.
    * **Do not drop out silently.** A round sums over all sites. A site
      that quietly returns nothing changes the objective function
      between optimizer iterations, so the fit converges to something
      that is not the estimand with no error raised anywhere.
    * **Be deterministic in theta.** Optimizers estimate gradients by
      finite differences, so a service that re-samples or jitters its
      answer turns the gradient into noise. Determinism also makes
      retries safe.
    * **Budget timeouts against call count.** A single fit may query
      every site hundreds of times.
    * **With a ThresholdMaster, availability is not optional.**
      Decryption is n-of-n, so an unreachable site withholds a partial
      and the round cannot be decrypted at all. Under a
      :class:`CKKSMaster` an unavailable site costs you a summand;
      under threshold keys it costs you the entire result.

    What the package leaves to you: transport, identity,
    authentication, attestation, remote key storage, serialization of
    the parameter bundle, retry and timeout policy — and any defense
    against a party that deviates from the protocol rather than merely
    observing it. The trust model throughout is honest-but-curious.
    """

    _needs_local_fn = False

    def set_public_params(self, params: PublicParams) -> RemoteSite:
        """Refused on the base class: setup must reach the endpoint.

        Storing the parameters here would configure this proxy and not
        the endpoint, which would then look wired while never having
        been told anything — a setup failure that surfaces much later,
        as a wrong answer. Missing remote setup fails closed instead.
        """
        raise NotImplementedError(
            f"RemoteSite {self.name!r} has no set_public_params(). Storing "
            "the parameters here would configure this proxy and not the "
            "endpoint, which would then look wired while never having been "
            "told anything. Implement it for your subclass: send params to "
            "the far end and have it retain them. Only public material "
            "travels -- a crypto context and a public key."
        )

    @abstractmethod
    def contribute(self, theta: Any) -> Ct | None:
        """Ask the far end for its encrypted contribution at ``theta``."""

    def keygen_round(self, ctx: Context, prev_pk: Any = None) -> Any:
        """Refused on the base class: the share must be made remotely.

        Running the inherited method would generate the share in *this*
        process, which is the thing threshold keys exist to prevent.
        """
        raise NotImplementedError(
            f"RemoteSite {self.name!r} has no keygen_round(). The share must "
            "be generated at the far end and stay there; running the "
            "inherited method would generate it in this process, which is "
            "the thing threshold keys exist to prevent. Implement it: send "
            "prev_pk, have the far end generate and retain its share, and "
            "return the cumulative public key."
        )

    def partial_decrypt(self, ciphertext: Any, lead: bool = False) -> Any:
        """Refused on the base class: the share must not travel."""
        raise NotImplementedError(
            f"RemoteSite {self.name!r} has no partial_decrypt(). Send the "
            "ciphertext to the far end, have it apply its own share, and "
            "return the partial. The share must not travel."
        )


class Master:
    """Abstract master: drives the protocol.

    Concrete subclasses implement :meth:`decrypt` and
    :attr:`_public_params`. There is deliberately no ``encrypt``:
    every party encrypts its own values with the public bundle it was
    handed at setup, through
    :meth:`~homomorphepy.params.PublicParams.encrypt`. The asymmetry is
    the point — decryption is privileged, encryption is not.
    """

    def __init__(self, name: str, ctx: Context):
        if not isinstance(ctx, Context):
            raise TypeError(
                "Master needs a homomorphepy Context (which records its "
                "scheme); build one with homomorphepy.fhe_context()"
            )
        self.name = _check_name(name)
        self.ctx = ctx
        self.workers: list[Site] = []

    @property
    def codec(self):
        return packed_codec(self.ctx)

    @property
    def _public_params(self) -> PublicParams:
        """The setup bundle this master hands out, once, at wiring time.

        Deliberately private. A site is autonomous once configured: it
        holds what it was given and encrypts with that. Reaching here
        at encryption time would be a party asking a coordinator for
        something it already has, which is the coupling the actor split
        exists to remove. The callers are :meth:`set_workers`,
        :func:`make_threshold_master`, and the checks in
        :meth:`aggregate` and :meth:`decrypt`.

        A party that needs the bundle asks a *site* for it, through
        :attr:`Site.params`, which involves no coordinator.
        """
        raise NotImplementedError

    def decrypt(self, ct: Any, length: int = 1) -> Any:  # pragma: no cover
        raise NotImplementedError

    # -- topology -----------------------------------------------------

    def set_workers(self, workers: Sequence[Site]) -> Master:
        """Wire workers to this master and send each the setup message.

        Use this for the master/worker (star) topology that distcomp-
        and DataSHIELD-style federated analyses follow.

        A :class:`ThresholdMaster` does **not** use this: its joint
        public key does not exist until key generation has run through
        every site, so :func:`make_threshold_master` takes the sites
        and returns a master already wired to them, in the order the
        chain fixed.
        """
        workers = list(workers)
        if not workers:
            raise ValueError("need at least one worker")
        params = self._public_params
        # Publish before recording, so a worker that refuses the setup
        # message -- a RemoteSite with no provisioning -- leaves the
        # master unwired rather than half-wired. It goes through the
        # method because a RemoteSite has to be told at the far end,
        # and writing into its proxy here would make it look configured
        # when it is not.
        for w in workers:
            w.set_public_params(params)
        self.workers = workers
        return self

    # -- protocol -----------------------------------------------------

    def aggregate(self, theta: Any, length: int = 1) -> Any:
        """One round of the master/worker protocol.

        The master broadcasts ``theta`` — and only ``theta``; each
        worker supplies its own data. Each worker returns an *already
        encrypted* contribution, the master sums them homomorphically
        and decrypts the total. No individual site's cleartext value
        reaches the master.

        Two failure modes, deliberately distinct. A worker that returns
        ``None`` found ``theta`` non-evaluable, and this returns NaN,
        which optimizers read as "back off and try elsewhere". A worker
        that raises :class:`SiteUnavailable` could not be reached at
        all; that propagates and aborts the round, because continuing
        would sum over a different set of sites and silently change the
        objective between iterations.

        NaN rather than None is deliberate: scipy's optimizers raise
        ``TypeError`` on a None objective, whereas NaN at least
        propagates. Callers driving an optimizer should still wrap this
        with an explicit penalty -- see the note in D6 and the
        finite-difference caveat in the plan.

        Non-evaluability is the one thing that travels in the clear,
        since CKKS cannot represent it. A master that chooses ``theta``
        adaptively therefore learns which parameter values break which
        site — a residual side channel no amount of encryption here
        removes.
        """
        if not self.workers:
            raise RuntimeError("master has no workers; call set_workers() first")

        params = self._public_params
        encrypted = []
        for w in self.workers:
            try:
                ct = w.contribute(theta)
            except SiteUnavailable as exc:
                # Carry the name, not the site: a site drags its data
                # and its key share into anything that logs this.
                raise SiteUnavailable(
                    f"site {w.name!r}: {exc}", site_name=w.name
                ) from exc
            if ct is None:
                return math.nan
            # Check what came back before adding it to the total. The
            # contract says a reply is a ciphertext or None; an
            # implementation that returned the plain number instead
            # would otherwise be summed in silently and the round would
            # report the right answer, having been handed the one
            # quantity the protocol exists to hide.
            params.check_encrypted(ct, "aggregate", who=w.name)
            encrypted.append(ct)

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
    def _public_params(self) -> PublicParams:
        return OpenFHEParams(self.ctx, self.keypair.publicKey)

    def decrypt(self, ct: Any, length: int = 1) -> Any:
        self._public_params.check_encrypted(ct, "decrypt")
        pt = self.ctx.Decrypt(unwrap(ct), self.keypair.secretKey)
        vals = self.codec.decode(pt, length)
        return vals[0] if length == 1 else vals


class ThresholdMaster(Master):
    """n-of-n threshold master; the sites hold the shares.

    The joint public key is built by chaining ``MultipartyKeyGen``
    across sites: the first generates ``(pk_1, sk_1)``, and each
    subsequent site ``i`` derives ``(pk_{1..i}, sk_i)`` from
    ``pk_{1..i-1}``. Encryption is under the final joint key.
    Decryption needs every site's partial, which the master fuses.

    **The master has no secret-key or secret-share attribute, and its
    methods use no secret material.** Its state is the crypto context
    and the joint public key, both public; the shares live at the sites
    that generated them and never travel.

    Read that at the right scope. In a single-process demonstration
    every role still inhabits one process, and the master holds the
    site objects in order to query them, so the shares are reachable
    from the master's object graph even though no attribute of the
    master contains one. A boundary between the parties requires
    separately controlled processes behind :class:`RemoteSite`.
    """

    def __init__(
        self,
        name: str,
        ctx: Context,
        joint_public_key: Any,
        sites: Sequence[Site],
    ):
        super().__init__(name, ctx)
        self.joint_public_key = joint_public_key
        self.workers = list(sites)

    @property
    def _public_params(self) -> PublicParams:
        return OpenFHEParams(self.ctx, self.joint_public_key)

    @property
    def sites(self) -> list[Site]:
        return list(self.workers)

    def set_workers(self, workers: Sequence[Site]) -> Master:
        """Refused: the key-generation chain already fixed the order.

        The site order the chain fixed is the order partial decryptions
        must fuse in, and re-wiring would break it.
        """
        raise RuntimeError(
            "a ThresholdMaster is already wired to its sites. Pass the sites "
            "to make_threshold_master(); the joint key is built from them, "
            "and the site order it fixes is the order partial decryptions "
            "fuse in."
        )

    def decrypt(self, ct: Any, length: int = 1) -> Any:
        """Collect partial decryptions from every site and fuse them.

        The master has no key material. It sends the ciphertext to each
        site and gets a partial back; the share that produced the
        partial never leaves the site. In a deployment each of these is
        a network round trip, which is why it goes through a method a
        :class:`RemoteSite` can implement.

        The lead/main distinction is a protocol role assigned by
        position in the key-generation chain, so the master tells each
        site which it is playing rather than the site deciding.
        Fusion needs only the context, so the master can do it: it is a
        public operation on public data.

        A site that returns a well-formed but wrong partial corrupts
        the result here with no error raised anywhere; see the warning
        on :func:`make_threshold_master`.
        """
        sites = self.workers
        if len(sites) < 2:
            raise RuntimeError("threshold master is not wired to its sites")
        self._public_params.check_encrypted(ct, "decrypt")

        partials = []
        for i, s in enumerate(sites):
            try:
                partials.append(s.partial_decrypt(ct, lead=(i == 0)))
            except SiteUnavailable as exc:
                raise SiteUnavailable(
                    f"site {s.name!r} did not return a partial decryption; "
                    "threshold decryption is n-of-n, so one missing partial "
                    "loses the whole result",
                    site_name=s.name,
                ) from exc

        pt = self.ctx.MultipartyDecryptFusion(partials)
        vals = self.codec.decode(pt, length)
        return vals[0] if length == 1 else vals


# ---- constructors ---------------------------------------------------------


def make_site(name: str, data: Any, local_fn: LocalFn) -> Site:
    """Construct a :class:`Site`."""
    return Site(name, data, local_fn)


def make_worker(name: str, data: Any, local_fn: LocalFn) -> Site:
    """Alias of :func:`make_site`, for master/worker phrasing.

    The site it returns serves either protocol: wire it to a
    :class:`CKKSMaster` with :meth:`Master.set_workers`, or hand it to
    :func:`make_threshold_master` and it generates and keeps a share.
    """
    return make_site(name, data, local_fn)


def make_ckks_master(name: str, ctx: Context, keypair: Any) -> CKKSMaster:
    """Construct a single-decrypter :class:`CKKSMaster`.

    The context must be a CKKS one. A ``CKKSMaster`` over BFV or BGV
    would work arithmetically, but every sentence of its documentation
    and the class name a user reads in printed output would be wrong
    about which scheme is in use. Exact-integer work goes through
    :func:`make_threshold_master`, which is scheme-agnostic by design
    and says so.
    """
    if not isinstance(ctx, Context):
        raise TypeError(
            "make_ckks_master needs a homomorphepy Context (which records "
            "its scheme); build one with homomorphepy.fhe_context()"
        )
    if ctx.scheme is not Scheme.CKKS:
        raise ValueError(
            f"make_ckks_master needs a CKKS context; this one is "
            f"{ctx.scheme.value}. For exact-integer work use "
            "make_threshold_master(), which reads the scheme from the "
            "context."
        )
    return CKKSMaster(name, ctx, keypair)


def make_threshold_master(
    name: str,
    ctx: Context,
    sites: Sequence[Site],
) -> ThresholdMaster:
    """Run the chained key generation and wire up a threshold protocol.

    Drives the ceremony *through the sites*: the lead generates a fresh
    keypair, and each subsequent site derives its own share from its
    predecessor's cumulative public key. Each step runs at the site,
    through :meth:`Site.keygen_round`, which keeps the share and
    returns only the cumulative *public* key. No share is generated
    centrally and none is returned here, so the master cannot hold one
    even by accident. Only public keys travel between parties, which is
    exactly what can be sent over a wire to an untrusted peer.

    Requires at least two **distinct, unconfigured** sites. Listing one
    site twice, or reusing a site that already holds a share or public
    parameters, is an error: the repeat would discard what the first
    round left behind, and under BFV or BGV nothing afterwards detects
    the loss.

    A ceremony that fails part-way — an unimplemented
    :class:`RemoteSite`, an unreachable endpoint, a context without
    ``MULTIPARTY`` — leaves no trace on the sites it had already
    visited: their shares and parameters are cleared before the error
    propagates, so the same sites can be used again once the cause is
    fixed. For a :class:`RemoteSite` that undo reaches the local proxy
    only, so a remote implementation should tolerate a repeated
    ceremony.

    ``MULTIPARTY`` is enabled here rather than demanded of the caller:
    a threshold master categorically needs it, so forgetting would be a
    pure footgun whose only symptom is an OpenFHE C++ error several
    calls later. ``Enable`` is idempotent, so passing
    ``features=[PKESchemeFeature.MULTIPARTY]`` to ``fhe_context()`` as
    well remains correct.

    What this does not defend against
    ---------------------------------
    The construction assumes participants follow the protocol
    (honest-but-curious). A site that deviates can return a well-formed
    ciphertext that is not its honest contribution, return a malformed
    partial decryption — which corrupts the fused plaintext *silently*,
    nothing in the scheme detects it — or contribute a degenerate share
    during key generation, weakening the threshold. The chain is
    sequential, so each site also sees its predecessors' cumulative
    public key; OpenFHE's multiparty key generation carries no proofs
    of knowledge or commitments, so rogue-key behavior is not prevented
    here. Defending against any of this needs verifiable decryption and
    committed key generation, neither of which this package provides.
    """
    from homomorphepy._backend import backend

    sites = list(sites)
    if len(sites) < 2:
        raise ValueError("threshold key generation requires at least two sites")

    for i, s in enumerate(sites):
        if not isinstance(s, Site):
            raise TypeError(f"sites[{i}] is not a Site")
        for j in range(i):
            # The same site twice is not two parties. Its second round
            # overwrites the share its first round generated, so the
            # joint key depends on a share nobody holds and every later
            # decryption is wrong -- silently, under BFV and BGV, which
            # have no approximation check to trip over.
            if s is sites[j]:
                raise ValueError(
                    f"sites {j} and {i} are the same party. Threshold key "
                    f"generation needs {len(sites)} distinct parties; a "
                    "repeated one overwrites the share it generated the "
                    "first time, and nothing detects the loss afterwards."
                )
            # Not a protocol failure, but it makes every later message
            # about a named site ambiguous.
            if s.name == sites[j].name:
                raise ValueError(f"sites {j} and {i} share the name {s.name!r}")
        if s.has_share or s._params is not None:
            raise ValueError(
                f"site {s.name!r} is already taking part in a protocol. A "
                "key-generation ceremony starts from unconfigured sites: "
                "joining a second one would discard the share and the "
                "parameters the first left behind. Build a fresh site with "
                "make_worker()."
            )

    ctx.Enable(backend().PKESchemeFeature.MULTIPARTY)

    # The chain runs site to site. Each call returns a public key and
    # nothing else; the share stays where it was generated. If any step
    # fails, the sites already visited hold a share belonging to a
    # ceremony that will never complete, and the checks above would then
    # refuse them a retry -- so undo the visit rather than leave it.
    touched: list[Site] = []
    try:
        pk = None
        for s in sites:
            touched.append(s)
            pk = s.keygen_round(ctx, pk)

        master = ThresholdMaster(name, ctx, pk, sites)
        # Everyone encrypts under the joint key, so the public bundle
        # goes back out to every site once the chain has completed.
        # This is the setup message, and it goes through the method so
        # that a remote party can receive it at the far end.
        params = master._public_params
        for s in sites:
            s.set_public_params(params)
    except BaseException:
        for s in touched:
            s._clear()
        raise

    return master


def make_joint_rotation_keys(
    master: ThresholdMaster,
    indices: Sequence[int],
) -> None:
    """Run the n-of-n ceremony that authorizes slot rotations.

    Rotating the slots of an encrypted vector needs a *rotation key* per
    index. Under a single key those come straight from the secret key;
    under threshold keys no such key exists, so the sites build them
    jointly, mirroring the ceremony that built the public key: the lead
    generates its share, each remaining site folds its own share in, and
    the accumulated map is registered against the joint public key's
    tag.

    Needed by any protocol that sums across slots or applies a matrix to
    an encrypted vector, since both are built from rotations.

    Two of the four calls involved are *static* on the OpenFHE
    ``CryptoContext`` rather than instance methods
    (``GetEvalAutomorphismKeyMap`` and ``InsertEvalAutomorphismKey``).
    Calling them on the instance appears to work and silently operates
    on the wrong registry, so they are reached through the class here.

    Returns nothing: the keys live in the context's registry, and every
    later ``EvalRotate`` under the joint key finds them there.
    """
    sites = master.sites
    if len(sites) < 2:
        raise ValueError("joint rotation keys need at least two sites")
    indices = [int(i) for i in indices]
    if not indices:
        raise ValueError("no rotation indices requested")

    cc = master.ctx.cc
    cls = type(cc)
    joint_tag = master.joint_public_key.GetKeyTag()

    # Lead site: seed the registry under its own key tag.
    lead = sites[0]
    cc.EvalRotateKeyGen(lead.secret_share, indices)
    running = cls.GetEvalAutomorphismKeyMap(lead.secret_share.GetKeyTag())

    # Remaining sites fold their shares in, one at a time.
    for site in sites[1:]:
        share = cc.MultiEvalAtIndexKeyGen(
            site.secret_share, running, indices, joint_tag
        )
        running = cc.MultiAddEvalAutomorphismKeys(running, share, joint_tag)

    cls.InsertEvalAutomorphismKey(running, joint_tag)
