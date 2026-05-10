"""
Module-level interoperability functions.

Each function delegates to the active backend cached in this module's ``_BACKEND``
slot. The slot is rebound by :func:`decent_bench.utils.interoperability_2.set_backend`
via :func:`_set_active_backend`. Calling any of these before ``set_backend`` raises
:class:`RuntimeError` via the sentinel's ``__getattr__``.

When this module and ``Backend`` are mypyc-compiled in the same group,
``_BACKEND.add(...)`` dispatches as a native compiled-to-compiled call — no Python
attribute lookup, no bound-method allocation per call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from decent_array.interoperability._backend_manager import register_backend_listener

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from decent_array.array import Array
    from decent_array.interoperability._abstracts import Backend
    from decent_array.types import ArrayKey, SupportedDevices

_BACKEND_INSTANCE: Backend | None = None
_error = RuntimeError("No backend active: call 'set_backend' with a supported framework to activate one.")


def _update_backend(backend: Backend | None) -> None:
    global _BACKEND_INSTANCE  # noqa: PLW0603
    _BACKEND_INSTANCE = backend


register_backend_listener(_update_backend)

# Array creation


def zeros(shape: tuple[int, ...]) -> Array:
    """Create an array of zeros with the given shape."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.zeros(shape)


def zeros_like(array: Array) -> Array:
    """Create an array of zeros matching the shape and type of ``array``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.zeros_like(array)


def ones(shape: tuple[int, ...]) -> Array:
    """Create an array of ones with the given shape."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.ones(shape)


def ones_like(array: Array) -> Array:
    """Create an array of ones matching the shape and type of ``array``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.ones_like(array)


def eye(n: int) -> Array:
    """Create an ``n x n`` identity matrix."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.eye(n)


def eye_like(array: Array) -> Array:
    """Create an identity matrix matching the trailing 2 dims of ``array``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.eye_like(array)


def device_to_native(device: SupportedDevices) -> Any:  # noqa: ANN401
    """Convert :class:`~decent_array.types.SupportedDevices` to the active backend's native device."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.device_to_native(device)


def device_of(array: Array) -> SupportedDevices:
    """Return the :class:`~decent_array.types.SupportedDevices` of ``array``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.device_of(array)


# Array manipulation


def copy(array: Array) -> Array:
    """Return a copy of ``array``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.copy(array)


def to_numpy(array: Array) -> NDArray[Any]:
    """Convert ``array`` to a NumPy array on CPU."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.to_numpy(array)


def from_numpy(array: NDArray[Any]) -> Array:
    """Convert a NumPy array on CPU to an :class:`~decent_array.Array` on the active backend."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.from_numpy(array)


def to_array(array: float | bool) -> Array:
    """Convert a Python scalar to an :class:`~decent_array.Array` on the active backend."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.to_array(array)


def stack(arrays: Sequence[Array], dim: int = 0) -> Array:
    """Stack a sequence of arrays along a new dimension."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.stack(arrays, dim)


def reshape(array: Array, shape: tuple[int, ...]) -> Array:
    """Reshape ``array`` to ``shape``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.reshape(array, shape)


def transpose(array: Array, dim: tuple[int, ...] | None = None) -> Array:
    """Transpose ``array``; ``None`` reverses dimensions."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.transpose(array, dim)


def shape(array: Array) -> tuple[int, ...]:
    """Return the shape of ``array``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.shape(array)


def size(array: Array) -> int:
    """Return the total number of elements in ``array``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.size(array)


def ndim(array: Array) -> int:
    """Return the number of dimensions of ``array``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.ndim(array)


def squeeze(array: Array, dim: int | tuple[int, ...] | None = None) -> Array:
    """Remove single-dimensional entries from ``array``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.squeeze(array, dim)


def unsqueeze(array: Array, dim: int) -> Array:
    """Insert a singleton dimension at ``dim``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.unsqueeze(array, dim)


def diag(array: Array) -> Array:
    """Diagonal: build from a vector or extract from a matrix."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.diag(array)


def astype(array: Array, dtype: type[float | int | bool]) -> float | int | bool:
    """Cast a single-element ``array`` to a Python scalar of ``dtype``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.astype(array, dtype)


# Linalg


def dot(array1: Array, array2: Array) -> Array:
    """Dot product of two arrays."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.dot(array1, array2)


def matmul(array1: Array, array2: Array) -> Array:
    """Matrix multiplication of two arrays."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.matmul(array1, array2)


def norm(
    array: Array,
    p: float = 2,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Norm of ``array``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.norm(array, p, dim, keepdims)


# Math reductions


def sum(  # noqa: A001
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Sum elements of ``array`` along ``dim``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.sum(array, dim, keepdims)


def mean(
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Mean of ``array`` along ``dim``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.mean(array, dim, keepdims)


def min(  # noqa: A001
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Minimum of ``array`` along ``dim``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.min(array, dim, keepdims)


def max(  # noqa: A001
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Maximum of ``array`` along ``dim``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.max(array, dim, keepdims)


def any(array: Array) -> bool:  # noqa: A001
    """Return True if any element of ``array`` is truthy."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.any(array)


def all(array: Array) -> bool:  # noqa: A001
    """Return True if all elements of ``array`` are truthy."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.all(array)


# Math elementwise


def add(array1: Array | float, array2: Array | float) -> Array:
    """Element-wise addition."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.add(array1, array2)


def iadd[T: Array](array1: T, array2: Array | float) -> T:
    """In-place element-wise addition."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.iadd(array1, array2)


def sub(array1: Array | float, array2: Array | float) -> Array:
    """Element-wise subtraction."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.sub(array1, array2)


def isub[T: Array](array1: T, array2: Array | float) -> T:
    """In-place element-wise subtraction."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.isub(array1, array2)


def mul(array1: Array | float, array2: Array | float) -> Array:
    """Element-wise multiplication."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.mul(array1, array2)


def imul[T: Array](array1: T, array2: Array | float) -> T:
    """In-place element-wise multiplication."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.imul(array1, array2)


def div(array1: Array | float, array2: Array | float) -> Array:
    """Element-wise division."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.div(array1, array2)


def idiv[T: Array](array1: T, array2: Array | float) -> T:
    """In-place element-wise division."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.idiv(array1, array2)


def pow(array: Array, p: float) -> Array:  # noqa: A001
    """Raise ``array`` to power ``p``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.pow(array, p)


def negative(array: Array) -> Array:
    """Element-wise negation."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.negative(array)


def absolute(array: Array) -> Array:
    """Element-wise absolute value."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.absolute(array)


def sqrt(array: Array) -> Array:
    """Element-wise square root."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.sqrt(array)


# Operators


def sign(array: Array) -> Array:
    """Element-wise sign."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.sign(array)


def maximum(array1: Array | float, array2: Array | float) -> Array:
    """Element-wise maximum."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.maximum(array1, array2)


def argmax(array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
    """Index of maximum value along ``dim``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.argmax(array, dim, keepdims)


def argmin(array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
    """Index of minimum value along ``dim``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.argmin(array, dim, keepdims)


def set_item(array: Array, key: ArrayKey, value: Array) -> None:
    """Set ``array[key] = value`` in place."""
    if _BACKEND_INSTANCE is None:
        raise _error
    _BACKEND_INSTANCE.set_item(array, key, value)


def get_item(array: Array, key: ArrayKey) -> Array:
    """Return ``array[key]``."""
    if _BACKEND_INSTANCE is None:
        raise _error
    return _BACKEND_INSTANCE.get_item(array, key)
