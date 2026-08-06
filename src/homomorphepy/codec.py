"""Scheme-appropriate plaintext encoding, selected from the context.

This is the direct analogue of homomorpheR's ``.packed_codec()``
(``R/sites.R:300``). The protocol body — encrypt local summaries, add
them homomorphically, decrypt the total — is identical across schemes;
only the encode/decode pair differs. CKKS carries reals and decodes to
approximate doubles, BFV and BGV carry exact integers.

In R the crypto context answers ``get_scheme_id()``, so a master never
has to be told which scheme it is driving. openfhe-python does not bind
that (P14), which is why :class:`~homomorphepy.context.Context` records
the scheme and this module reads it from there.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from homomorphepy.context import Context, Scheme

__all__ = ["Codec", "packed_codec"]


@dataclass(frozen=True)
class Codec:
    """An encode/decode pair matched to a context's scheme."""

    scheme: Scheme
    _ctx: Context

    def encode(self, values: Sequence[float] | float) -> Any:
        """Pack values into a plaintext appropriate to the scheme."""
        vals = [values] if isinstance(values, (int, float)) else list(values)
        cc = self._ctx.cc
        if self.scheme.is_approximate:
            return cc.MakeCKKSPackedPlaintext([float(v) for v in vals])
        # BFV/BGV are integer schemes; coerce exactly as R's codec does
        # (sites.R:437 uses as.integer), so a caller passing 2.0 gets 2
        # rather than a silent encoding error.
        return cc.MakePackedPlaintext([int(v) for v in vals])

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
