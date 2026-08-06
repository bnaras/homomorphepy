"""Crypto contexts that remember their own scheme.

openfhe-python does not bind ``CryptoContext::GetSchemeId`` (upstream
defect P14), so a bare context cannot be asked what scheme it was built
for. The codec layer relies on being able to ask: ``packed_codec()``
branches on the scheme to pick the encode/decode pair, which is what
lets one threshold master drive CKKS real-valued work and BFV/BGV
exact-integer work with the context as the single source of truth.

:class:`Context` restores that property by recording the scheme at
construction. The cost, noted in decision D6, is that the *wrapper*
rather than the context becomes the source of truth — so examples must
build contexts through :func:`fhe_context` rather than calling
``GenCryptoContext`` directly.

``PKE``, ``KEYSWITCH`` and ``LEVELEDSHE`` are enabled by default, since
every protocol here needs them; anything else is passed explicitly.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from homomorphepy._backend import backend

__all__ = ["Scheme", "Context", "fhe_context"]


class Scheme(str, Enum):
    """The schemes homomorphepy builds contexts for."""

    BFV = "BFV"
    BGV = "BGV"
    CKKS = "CKKS"

    @property
    def is_approximate(self) -> bool:
        """True for CKKS, whose arithmetic carries approximation error.

        The distinction drives both encoding (real vs integer packing)
        and comparison: BFV and BGV results are asserted with ``==``,
        CKKS results against a tolerance.
        """
        return self is Scheme.CKKS


class Context:
    """A CryptoContext that knows its scheme.

    Wraps the raw ``openfhe.CryptoContext``. Attribute access falls
    through, so every method the extension exposes remains reachable:
    ``ctx.EvalRotate(...)`` works exactly as on the bare object.
    """

    __slots__ = ("_cc", "_scheme", "_params")

    def __init__(self, cc: Any, scheme: Scheme, params: dict[str, Any] | None = None):
        self._cc = cc
        self._scheme = Scheme(scheme)
        self._params = dict(params or {})

    @property
    def cc(self) -> Any:
        """The underlying ``openfhe.CryptoContext``."""
        return self._cc

    @property
    def scheme(self) -> Scheme:
        """The scheme this context was built for (see P14)."""
        return self._scheme

    @property
    def params(self) -> dict[str, Any]:
        """The parameters this context was constructed with.

        Recorded so examples can print the full tuple, which is how
        drift in the parameters a context was built with shows up rather
        than as an unexplained numeric difference.
        """
        return dict(self._params)

    @property
    def ring_dimension(self) -> int:
        return int(self._cc.GetRingDimension())

    def __getattr__(self, name: str) -> Any:
        # __slots__ means this only fires for names not on the wrapper.
        return getattr(self._cc, name)

    def __repr__(self) -> str:
        shown = ", ".join(f"{k}={v}" for k, v in sorted(self._params.items()))
        return (
            f"<Context {self._scheme.value} ring_dim={self.ring_dimension}"
            f"{': ' + shown if shown else ''}>"
        )


# Argument names are the snake_case of the C++ CCParams setters, so a
# parameter can be looked up directly in the OpenFHE headers.
_SETTERS = {
    "multiplicative_depth": "SetMultiplicativeDepth",
    "scaling_mod_size": "SetScalingModSize",
    "first_mod_size": "SetFirstModSize",
    "batch_size": "SetBatchSize",
    "ring_dim": "SetRingDim",
    "plaintext_modulus": "SetPlaintextModulus",
    "security_level": "SetSecurityLevel",
    "scaling_technique": "SetScalingTechnique",
    "key_switch_technique": "SetKeySwitchTechnique",
    "multiparty_mode": "SetMultipartyMode",
    "digit_size": "SetDigitSize",
    "num_large_digits": "SetNumLargeDigits",
    "max_relin_sk_deg": "SetMaxRelinSkDeg",
    "threshold_num_of_parties": "SetThresholdNumOfParties",
}


def fhe_context(
    scheme: str | Scheme,
    *,
    features: list[Any] | None = None,
    **params: Any,
) -> Context:
    """Create a :class:`Context` for ``scheme``.

    ``PKE``, ``KEYSWITCH`` and ``LEVELEDSHE`` are enabled automatically,
    automatically; ``features`` adds to that triple
    rather than replacing it. Pass ``MULTIPARTY`` there for threshold
    protocols.

    Parameters are given in snake_case and forwarded to the matching
    ``CCParams`` setter::

        ctx = fhe_context("CKKS", multiplicative_depth=1,
                          scaling_mod_size=59, first_mod_size=60,
                          batch_size=8)

    Raises ``TypeError`` on an unknown parameter name rather than
    silently ignoring it — a dropped parameter would change the
    security level or the noise budget with no visible symptom.
    """
    ofhe = backend()
    scheme = Scheme(scheme)

    ctor = {
        Scheme.BFV: ofhe.CCParamsBFVRNS,
        Scheme.BGV: ofhe.CCParamsBGVRNS,
        Scheme.CKKS: ofhe.CCParamsCKKSRNS,
    }[scheme]
    p = ctor()

    unknown = sorted(set(params) - set(_SETTERS))
    if unknown:
        raise TypeError(
            f"unknown parameter(s) for fhe_context: {unknown}. "
            f"Known: {sorted(_SETTERS)}"
        )

    for name, value in params.items():
        setter = getattr(p, _SETTERS[name], None)
        if setter is None:
            raise TypeError(
                f"{scheme.value} contexts do not accept {name!r} "
                f"(no {_SETTERS[name]} on {type(p).__name__})"
            )
        setter(value)

    cc = ofhe.GenCryptoContext(p)

    feat = ofhe.PKESchemeFeature
    for f in (feat.PKE, feat.KEYSWITCH, feat.LEVELEDSHE):
        cc.Enable(f)
    for f in features or ():
        cc.Enable(f)

    return Context(cc, scheme, params)
