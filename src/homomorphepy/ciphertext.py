"""Ciphertext arithmetic through Python operators.

openfhe-python binds exactly one operator on ``Ciphertext``,
``__add__`` (upstream defect P12; ``bindings.cpp:1610-1614``, with the
``py::self`` line commented out). ``ct - ct``, ``ct * 2.0`` and ``-ct``
all raise ``TypeError``, and there is no ``__radd__``, so even
``sum(cts)`` fails on the initial ``0 + ct``.

Every operation is reachable as ``cc.EvalAdd`` / ``EvalSub`` /
``EvalMult`` / ``EvalNegate``, so this is an ergonomics gap rather than
a capability one. But ciphertext arithmetic is the hot path in every
protocol: R reads ``Reduce(`+`, cts)`` and ``ct * (1/N)`` because
``openfhe.R`` supplies the whole operator group through an S3 ``Ops``
handler. :class:`Ct` restores the same surface here, so ported code
reads like the R original instead of being transliterated into method
calls.

The dispatch table mirrors ``openfhe.R``'s
(``R/methods-eval.R:30-80``) — including the asymmetric cases, where
scalar-minus-ciphertext is computed as ``negate(sub(ct, scalar))``
because OpenFHE offers no reversed subtraction.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Ct", "wrap", "unwrap"]

_Scalar = (int, float)


def unwrap(x: Any) -> Any:
    """The raw ciphertext/plaintext behind a :class:`Ct`, if any."""
    return x.raw if isinstance(x, Ct) else x


def wrap(x: Any, cc: Any) -> Ct:
    """Wrap a raw ciphertext, or rewrap an existing :class:`Ct`."""
    return x if isinstance(x, Ct) else Ct(x, cc)


class Ct:
    """A ciphertext with Python arithmetic operators.

    Binds ``+``, ``-``, ``*`` and unary ``-`` over the context's
    ``Eval*`` methods, with reflected forms so ``sum()`` and
    ``functools.reduce(operator.add, ...)`` work. Comparison operators
    are deliberately absent: FHE ciphertexts are not ordered, and
    defining ``==`` to mean anything here would invite silent nonsense.

    The context travels with the ciphertext because every OpenFHE
    operation needs it, and threading it separately through protocol
    code is exactly the bookkeeping this wrapper exists to remove.
    """

    __slots__ = ("_ct", "_cc")

    def __init__(self, ct: Any, cc: Any):
        if isinstance(ct, Ct):  # defensive: double-wrapping is a bug
            ct = ct.raw
        self._ct = ct
        self._cc = cc

    # -- access -------------------------------------------------------

    @property
    def raw(self) -> Any:
        """The underlying ``openfhe.Ciphertext``."""
        return self._ct

    @property
    def cc(self) -> Any:
        """The context this ciphertext belongs to (raw, not wrapped)."""
        return self._cc

    def __getattr__(self, name: str) -> Any:
        # Ciphertext methods (GetLevel, GetSlots, ...) stay reachable.
        return getattr(self._ct, name)

    def _new(self, raw: Any) -> Ct:
        return Ct(raw, self._cc)

    # -- addition -----------------------------------------------------

    def __add__(self, other: Any) -> Ct:
        return self._new(self._cc.EvalAdd(self._ct, unwrap(other)))

    def __radd__(self, other: Any) -> Ct:
        # sum() starts from int 0; treat that as identity so
        # sum(cts) works without a start= argument.
        if isinstance(other, int) and other == 0:
            return self
        return self._new(self._cc.EvalAdd(self._ct, unwrap(other)))

    # -- subtraction --------------------------------------------------

    def __sub__(self, other: Any) -> Ct:
        return self._new(self._cc.EvalSub(self._ct, unwrap(other)))

    def __rsub__(self, other: Any) -> Ct:
        # OpenFHE has no reversed subtraction: scalar - ct is computed
        # as -(ct - scalar), matching openfhe.R's methods-eval.R:42-44.
        return self._new(self._cc.EvalNegate(self._cc.EvalSub(self._ct, unwrap(other))))

    # -- multiplication -----------------------------------------------

    def __mul__(self, other: Any) -> Ct:
        return self._new(self._cc.EvalMult(self._ct, unwrap(other)))

    def __rmul__(self, other: Any) -> Ct:
        return self._new(self._cc.EvalMult(self._ct, unwrap(other)))

    # -- unary --------------------------------------------------------

    def __neg__(self) -> Ct:
        return self._new(self._cc.EvalNegate(self._ct))

    def __pos__(self) -> Ct:
        return self

    # -- deliberately unsupported -------------------------------------

    def __eq__(self, other: object) -> bool:
        raise TypeError(
            "ciphertexts cannot be compared; decrypt first, then compare "
            "the plaintexts within a tolerance appropriate to the scheme"
        )

    __hash__ = None  # type: ignore[assignment]

    def __truediv__(self, other: Any) -> Ct:
        if isinstance(other, _Scalar):
            # Division by a public scalar is multiplication by its
            # reciprocal; anything else has no FHE meaning.
            return self._new(self._cc.EvalMult(self._ct, 1.0 / float(other)))
        raise TypeError(
            "ciphertext division is only defined by a public scalar; "
            "there is no homomorphic reciprocal"
        )

    def __repr__(self) -> str:
        try:
            lvl = self._ct.GetLevel()
        except Exception:  # pragma: no cover - depends on scheme/state
            lvl = "?"
        return f"<Ct level={lvl}>"
