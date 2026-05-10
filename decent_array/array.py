"""
Lightweight wrapper around backend-native arrays.

The :class:`Array` class wraps a single value of the active backend's framework type.
Under the single-active-backend invariant maintained by
:mod:`decent_array.interoperability.backend_manager`, every :class:`Array` at runtime
holds a value from the same framework, so operators dispatch directly to the active
backend without per-call isinstance dispatch.

Operator contract is *strict*: binary arithmetic and indexing accept either another
:class:`Array` or a Python scalar (``int``/``float``). Pass other framework-native
arrays through :func:`decent_array.iop.get_item` and friends, not through the
operator path.

Hot-path notes:

* ``__add__``/``__sub__``/``__mul__``/``__truediv__``/``__matmul__`` and the unary
  ``__neg__``/``__abs__``/``__pow__`` are inlined: every supported framework's tensor
  implements the equivalent operator natively with numpy-equivalent semantics, so
  routing through the backend saves nothing.
* Operators that *do* go through the backend (in-place math, indexing, properties
  like ``shape``/``transpose``) read the cached ``_backend`` slot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from decent_array.interoperability.backend_manager import register_backend_listener

if TYPE_CHECKING:
    from decent_array.interoperability.abstracts import _Backend
    from decent_array.types import ArrayKey


_BACKEND_INSTANCE: _Backend | None = None


def _update_backend(backend: _Backend | None) -> None:
    global _BACKEND_INSTANCE  # noqa: PLW0603
    _BACKEND_INSTANCE = backend


register_backend_listener(_update_backend)


class Array:  # noqa: PLR0904
    """
    Wrapper around a single backend-native array.

    Storage is two slots (``value``, ``_backend``) declared via ``__slots__``;
    instances have no ``__dict__``. Every operator that delegates to the backend
    reads the cached slot, so dispatch is one slot load plus the backend method call.
    """

    __slots__ = ("_backend", "value")

    def __init__(self, value: Any) -> None:  # noqa: ANN401
        """
        Wrap ``value`` in an :class:`Array`.

        Args:
            value: A backend-native array (or scalar) to wrap. The attribute is typed
                as :class:`typing.Any` because the wrapper is intentionally type-erased
                — backend code accesses :attr:`value` knowing the framework type, and
                typing it more strictly forces a ``cast`` at every call site without
                runtime benefit.

        Raises:
            RuntimeError: If no backend is registered yet. An :class:`Array` cannot be
                constructed until a backend is set; call :func:`set_backend` to initialize
                the interoperability layer.

        """
        if _BACKEND_INSTANCE is None:
            raise RuntimeError(
                "No backend registered yet. An Array cannot be constructed until a backend is set. "
                "Call set_backend() to initialize the interoperability layer."
            )

        self.value: Any = value
        self._backend: _Backend = _BACKEND_INSTANCE

    # Binary arithmetic ----------------------------------------------------

    def __add__(self, other: Array | float) -> Array:
        """Return the sum of the array and another array or a scalar."""
        return Array(self.value + (other.value if type(other) is Array else other))

    def __radd__(self, other: float) -> Array:
        """Return the sum of the array and a scalar."""
        return Array(other + self.value)

    def __sub__(self, other: Array | float) -> Array:
        """Return the subtraction of another array or a scalar from the array."""
        return Array(self.value - (other.value if type(other) is Array else other))

    def __rsub__(self, other: float) -> Array:
        """Return the subtraction of the array from a scalar."""
        return Array(other - self.value)

    def __mul__(self, other: Array | float) -> Array:
        """Return the product of the array and another array or a scalar."""
        return Array(self.value * (other.value if type(other) is Array else other))

    def __rmul__(self, other: float) -> Array:
        """Return the product of the array and a scalar."""
        return Array(other * self.value)

    def __truediv__(self, other: Array | float) -> Array:
        """Return the true division of the array by ``other``."""
        return Array(self.value / (other.value if type(other) is Array else other))

    def __rtruediv__(self, other: float) -> Array:
        """Return the true division of ``other`` by the array."""
        return Array(other / self.value)

    def __matmul__(self, other: Array) -> Array:
        """Return the matrix multiplication of the array with ``other``."""
        return Array(self.value @ other.value)

    def __rmatmul__(self, other: Array) -> Array:
        """Return the matrix multiplication of ``other`` with the array."""
        return Array(other.value @ self.value)

    def __pow__(self, other: float) -> Array:
        """Exponentiate the array by a scalar power."""
        # numpy/torch/jax/tf all implement ``tensor ** p`` with semantics matching the
        # backend's ``pow``; routing through the backend would cost an extra method
        # call for no behavioral difference.
        return Array(self.value**other)

    # In-place arithmetic --------------------------------------------------
    #
    # The backend handles the framework's mutability semantics: numpy/pytorch mutate
    # `value` in place, jax/tensorflow rebind it. In every case the returned object is
    # the same wrapper instance, so we just discard the return and yield ``self``.

    def __iadd__(self, other: Array | float) -> Self:
        """In-place addition."""
        self._backend.iadd(self, other)
        return self

    def __isub__(self, other: Array | float) -> Self:
        """In-place subtraction."""
        self._backend.isub(self, other)
        return self

    def __imul__(self, other: Array | float) -> Self:
        """In-place multiplication."""
        self._backend.imul(self, other)
        return self

    def __itruediv__(self, other: Array | float) -> Self:
        """In-place true division."""
        self._backend.idiv(self, other)
        return self

    # Unary ----------------------------------------------------------------

    def __neg__(self) -> Array:
        """Return the negation of the array."""
        # Native ``-tensor`` matches the backend's ``negative`` wrapper across all
        # supported frameworks, so the indirection is not needed.
        return Array(-self.value)

    def __abs__(self) -> Array:
        """Return the absolute value of the array."""
        # Same rationale as ``__neg__`` — native ``abs(tensor)`` matches each
        # backend's ``absolute`` implementation.
        return Array(abs(self.value))

    # Indexing -------------------------------------------------------------

    def __getitem__(self, key: ArrayKey) -> Array:
        """Return the item at ``key``."""
        return self._backend.get_item(self, key)

    def __setitem__(self, key: ArrayKey, value: Array | float) -> None:
        """Set the item at ``key`` to ``value``."""
        if type(value) is not Array:
            value = Array(value)
        self._backend.set_item(self, key, value)

    # Containers / iteration ----------------------------------------------

    def __len__(self) -> int:
        """Return the size of the first dimension of the array."""
        return len(self.value)

    # Coercion -------------------------------------------------------------

    def __float__(self) -> float:
        """Coerce a scalar array to a Python float."""
        return float(self._backend.astype(self, float))

    # Repr -----------------------------------------------------------------

    def __repr__(self) -> str:
        """Show the wrapper and the wrapped value."""
        return f"Array({self.value!r})"

    def __str__(self) -> str:
        """Stringify the wrapped value, not the wrapper."""
        return str(self.value)

    # Properties -----------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the array."""
        return self._backend.shape(self)

    @property
    def size(self) -> int:
        """Return the total number of elements in the array."""
        return self._backend.size(self)

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the array."""
        return self._backend.ndim(self)

    @property
    def transpose(self) -> Array:
        """Return a transposed view of the array."""
        return self._backend.transpose(self)

    @property
    def T(self) -> Array:  # noqa: N802
        """Return a transposed view of the array."""
        return self.transpose

    @property
    def any(self) -> bool:
        """Return True if any element of the array is truthy."""
        return self._backend.any(self)

    @property
    def all(self) -> bool:
        """Return True if all elements of the array are truthy."""
        return self._backend.all(self)
