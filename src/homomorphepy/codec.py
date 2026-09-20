"""Scheme-appropriate plaintext encoding, selected from the context.

The protocol body — encrypt local summaries, add them
homomorphically, decrypt the total — is identical across schemes; only
the encode/decode pair differs. CKKS carries reals and decodes to
approximate doubles, BFV and BGV carry exact integers.

Ideally the crypto context would answer "which scheme am I?" so a
master never has to be told. openfhe-python does not bind
``GetSchemeId`` (upstream defect P14), which is why
:class:`~homomorphepy.context.Context` records the scheme at
construction and this module reads it from there.

**The exact schemes refuse what they cannot carry.** An earlier version
of this module coerced with ``int(v)`` before packing, which turns 0.9
into 0 and reports a total that was never asked for. BFV and BGV have
no approximation error to trip over, so a rounded or wrapped value
comes back as a plausible integer with nothing raised anywhere. See
:func:`as_exact_integer`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from homomorphepy.context import Context, Scheme

__all__ = ["Codec", "Packable", "as_exact_integer", "packed_codec"]

# What can be packed into a plaintext: a scalar, an ordinary sequence,
# or a NumPy array. The array arm is not redundant -- ``np.ndarray`` is
# not a ``Sequence`` as far as a type checker is concerned, and an
# array is the common case in every example here.
Packable = float | int | Sequence[float] | np.ndarray

# MakePackedPlaintext takes int64_t. Values beyond it are a caller
# error rather than a modulus question, and worth a distinct message.
_INT64_MAX = 2**63 - 1


def as_exact_integer(values: Packable, ctx: Context) -> list[int]:
    """Coerce to integers for BFV/BGV, refusing anything that would lie.

    Four ways a value fails to be carryable, each of which is silent if
    it is merely coerced:

    * NaN or an infinity — ``int(float("nan"))`` raises, but ``None``
      and NumPy NaN reach here by different routes, so it is checked.
    * Non-integral — ``int(0.9)`` is 0. The sum then reports zero for
      something that was never zero.
    * Beyond int64 — outside what ``MakePackedPlaintext`` accepts.
    * At or above half the plaintext modulus — this one encodes and
      encrypts happily and only goes wrong on the first addition, which
      wraps. Two values of 40000 under ``t = 65537`` sum to 14463.

    OpenFHE itself rejects a value strictly above ``t``; the band from
    ``t/2`` to ``t`` is the gap this closes.
    """
    vals = [values] if isinstance(values, (int, float)) else list(values)
    out: list[int] = []
    for v in vals:
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            raise ValueError(
                "exact-integer schemes cannot represent NaN or an infinity; "
                "a site that cannot evaluate a query returns None from its "
                "local_fn, which contribute() handles before encryption"
            )
        if f != round(f):
            raise ValueError(
                f"{v!r} is not an integer. BFV and BGV carry exact integers, "
                "and rounding here would report a total that is not the one "
                "asked for; use a CKKS context for real-valued work"
            )
        i = int(round(f))
        if abs(i) > _INT64_MAX:
            raise ValueError(f"{v!r} is outside the range MakePackedPlaintext accepts")
        out.append(i)

    t = _plaintext_modulus(ctx)
    if t is not None:
        over = [i for i in out if abs(i) >= t // 2]
        if over:
            raise ValueError(
                f"{over[0]!r} does not fit the plaintext modulus {t}: it would "
                "wrap around and decrypt to a different number"
            )
    return out


def _plaintext_modulus(ctx: Context) -> int | None:
    """The plaintext modulus, for exact schemes only.

    ``GetPlaintextModulus()`` answers on a CKKS context too, with a
    number that means something else entirely (it returns the scaling
    mod size), so it is only consulted where it is meaningful.
    """
    if ctx.scheme.is_approximate:
        return None
    try:
        t = int(ctx.cc.GetPlaintextModulus())
    except Exception:  # noqa: BLE001 - absent or unreadable is not fatal
        return None
    return t if t > 0 else None


@dataclass(frozen=True)
class Codec:
    """An encode/decode pair matched to a context's scheme."""

    scheme: Scheme
    _ctx: Context

    def encode(self, values: Packable) -> Any:
        """Pack values into a plaintext appropriate to the scheme.

        The three supported schemes are named explicitly. Treating
        "anything that is not CKKS" as an integer scheme would mean a
        context this package has no codec for — a scheme added later,
        or one built by mistake — quietly encoding as packed integers
        instead of saying so.
        """
        cc = self._ctx.cc
        if self.scheme is Scheme.CKKS:
            vals = [values] if isinstance(values, (int, float)) else list(values)
            return cc.MakeCKKSPackedPlaintext([float(v) for v in vals])
        if self.scheme in (Scheme.BFV, Scheme.BGV):
            return cc.MakePackedPlaintext(as_exact_integer(values, self._ctx))
        raise TypeError(
            f"no plaintext codec for scheme {self.scheme!r}: homomorphepy "
            "carries CKKS for real-valued work and BFV or BGV for exact "
            "integers"
        )

    def decode(self, pt: Any, length: int | None = None) -> list[float] | list[int]:
        """Unpack a decrypted plaintext, truncated to ``length``."""
        if length is not None:
            pt.SetLength(int(length))
        if self.scheme.is_approximate:
            vals = list(pt.GetRealPackedValue())
        else:
            vals = list(pt.GetPackedValue())
        return vals if length is None else vals[: int(length)]


def packed_codec(ctx: Context) -> Codec:
    """The codec for ``ctx``, chosen from the scheme it recorded."""
    if not isinstance(ctx, Context):
        raise TypeError(
            "packed_codec needs a homomorphepy Context, not a bare "
            "CryptoContext: openfhe-python cannot report a context's "
            "scheme (upstream P14), so the scheme must come from the "
            "wrapper. Build contexts with homomorphepy.fhe_context()."
        )
    return Codec(scheme=ctx.scheme, _ctx=ctx)
