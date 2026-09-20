"""The public bundle a party is handed at setup, and encrypts with.

Setup is a *message*, not an assignment. A party receives its
:class:`PublicParams` once, when it is wired, and from then on computes
and encrypts entirely on its own: it holds no reference to whoever
aggregates its answers, and needs none. Reaching back to a coordinator
at encryption time is the coupling this module exists to remove.

What crosses is public in full — a crypto context and a public key.
There is no secret material in a :class:`PublicParams` and no attribute
for one to occupy, which is why :meth:`PublicParams.__repr__` prints
that fact rather than asserting it in prose.

**Encryption names no party.** :meth:`PublicParams.encrypt` is the one
encryption entry point in the package, and it takes only public
material. The asymmetry is worth reading off the API: decryption is
privileged — it needs secret material, or the standing to convene every
site — while encryption is available to anyone holding the bundle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from homomorphepy._backend import backend
from homomorphepy.ciphertext import Ct, unwrap
from homomorphepy.codec import Packable, packed_codec
from homomorphepy.context import Context

__all__ = [
    "BadContribution",
    "KeyMismatch",
    "OpenFHEParams",
    "PublicParams",
]


class BadContribution(TypeError):
    """A party replied with something that is not an encrypted value."""


class KeyMismatch(ValueError):
    """An encrypted value was produced under some other key."""


class PublicParams(ABC):
    """Abstract public setup bundle.

    Concrete subclasses carry whatever public material their backend
    needs. They must be able to answer :attr:`tag` — a public
    fingerprint of the key they encrypt under — and to encrypt.
    """

    @property
    @abstractmethod
    def tag(self) -> str:
        """A short public fingerprint of the key this bundle uses."""

    @abstractmethod
    def encrypt(self, value: Packable) -> Ct:
        """Encrypt ``value`` under these parameters."""

    @abstractmethod
    def check_encrypted(self, x: Any, what: str, who: str | None = None) -> Any:
        """Raise unless ``x`` is a ciphertext under this bundle's key."""


class OpenFHEParams(PublicParams):
    """Public parameters for the OpenFHE backends.

    The crypto context and the public key to encrypt under — the joint
    public key when the protocol uses threshold keys. Both are public.
    The scheme is read back from the context, so one class serves CKKS,
    BFV and BGV.
    """

    __slots__ = ("_ctx", "_pk")

    def __init__(self, ctx: Context, pk: Any):
        if not isinstance(ctx, Context):
            raise TypeError(
                "OpenFHEParams needs a homomorphepy Context (which records "
                "its scheme); build one with homomorphepy.fhe_context()"
            )
        self._ctx = ctx
        self._pk = pk

    @property
    def ctx(self) -> Context:
        return self._ctx

    @property
    def pk(self) -> Any:
        return self._pk

    @property
    def tag(self) -> str:
        """The public key's OpenFHE key tag.

        OpenFHE stamps every key and every ciphertext with a tag and
        preserves it through homomorphic operations, so comparing tags
        answers "was this produced under my key?" exactly, on public
        data, without decrypting anything. That one equality turns a
        whole class of wrong-key failures — a worker re-wired to a
        second master, a ciphertext from another protocol, a site's own
        share applied to a foreign value — from a silent wrong number
        into an error. Untrapped under BFV, a re-keyed decryption
        returns a plausible integer and nothing is raised anywhere.
        """
        return str(self._pk.GetKeyTag())

    def encrypt(self, value: Packable) -> Ct:
        """Encrypt ``value``, encoding as the context's scheme requires.

        Packed reals under CKKS, packed integers under BFV and BGV. The
        exact schemes reject a value they cannot represent rather than
        rounding it away; see
        :func:`~homomorphepy.codec.as_exact_integer`.
        """
        pt = packed_codec(self._ctx).encode(value)
        return Ct(self._ctx.cc.Encrypt(self._pk, pt), self._ctx.cc)

    def check_encrypted(self, x: Any, what: str, who: str | None = None) -> Any:
        """Raise unless ``x`` is a ciphertext under this bundle's key.

        Two failures, and the first is the one worth dwelling on.

        Without the *type* half, a site that answers in cleartext has
        its plain number folded into the running total by ordinary
        addition and the round returns the **correct** answer — with
        that site's individual contribution having crossed the boundary
        the protocol exists to keep it behind. Nothing else in the
        pipeline notices.

        The *tag* half catches a value produced under some other key.
        """
        ciphertext_type = backend().Ciphertext
        raw = unwrap(x)
        where = f" from site {who!r}" if who else ""
        if not isinstance(raw, ciphertext_type):
            raise BadContribution(
                f"cannot {what} a {type(x).__name__}{where}: an encrypted "
                "value is expected. A party that replies in cleartext hands "
                "over exactly the individual quantity the protocol keeps "
                "hidden, and the arithmetic would go through without "
                "complaint."
            )
        if str(raw.GetKeyTag()) != self.tag:
            raise KeyMismatch(
                f"encrypted value{where} was produced under a different key: "
                "its key tag does not match this protocol's public key. "
                "Decrypting it would return noise -- and under BFV or BGV "
                "that noise is a plausible integer, raising nothing."
            )
        return x

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OpenFHEParams) and self.tag == other.tag

    def __hash__(self) -> int:
        return hash(self.tag)

    def __repr__(self) -> str:
        return (
            f"<OpenFHEParams {self._ctx.scheme.value}\n"
            f"  public key  {self.tag}\n"
            f"  secret material: none>"
        )
