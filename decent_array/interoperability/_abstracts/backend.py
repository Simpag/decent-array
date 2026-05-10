"""
Abstract :class:`Backend` contract.

All abstract methods live in this single class rather than across six mixin ABCs. The
flat layout is mypyc-compatible: when this module is included in the same compilation
group as the concrete backends, ``_BACKEND.add(self, other)`` becomes a native
compiled-to-compiled call (no Python attribute lookup, no bound-method allocation),
which removes the need for a ``raw_ops`` escape hatch on the hot path.

Section dividers in this file group related operations the way the legacy split files
did (creation, manipulation, linalg, math, operators, RNG); the only thing that
changed is that they all live on one class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from decent_array.types import SupportedDevices

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from decent_array import Array
    from decent_array.types import ArrayKey


class Backend(ABC):  # noqa: PLR0904
    """
    Abstract base class for a backend.

    Concrete backends are bound to a single :class:`SupportedDevices` at construction
    time; that device is the default for all new arrays produced by this backend.
    """

    def __init__(self, device: SupportedDevices = SupportedDevices.CPU) -> None:
        self.device: SupportedDevices = device

    # Array creation ------------------------------------------------------

    @abstractmethod
    def zeros(self, shape: tuple[int, ...]) -> Array:
        """Create an array of zeros with the given shape."""

    @abstractmethod
    def zeros_like(self, array: Array) -> Array:
        """Create an array of zeros matching the shape and type of ``array``."""

    @abstractmethod
    def ones(self, shape: tuple[int, ...]) -> Array:
        """Create an array of ones with the given shape."""

    @abstractmethod
    def ones_like(self, array: Array) -> Array:
        """Create an array of ones matching the shape and type of ``array``."""

    @abstractmethod
    def eye(self, n: int) -> Array:
        """Create an ``n x n`` identity matrix."""

    @abstractmethod
    def eye_like(self, array: Array) -> Array:
        """Create an identity matrix matching the trailing two dims of ``array``."""

    @abstractmethod
    def device_to_native(self, device: SupportedDevices) -> Any:  # noqa: ANN401
        """Convert :class:`SupportedDevices` to the backend's native device representation."""

    @abstractmethod
    def device_of(self, array: Array) -> SupportedDevices:
        """Return the :class:`SupportedDevices` of the given array."""

    # Array manipulation --------------------------------------------------

    @abstractmethod
    def copy(self, array: Array) -> Array:
        """Return a copy of ``array``."""

    @abstractmethod
    def to_numpy(self, array: Array) -> NDArray[Any]:
        """Convert ``array`` to a NumPy array on the CPU."""

    @abstractmethod
    def from_numpy(self, array: NDArray[Any]) -> Array:
        """Convert a NumPy array on the CPU to an :class:`Array` on this backend."""

    @abstractmethod
    def from_numpy_like(self, array: NDArray[Any], like: Array) -> Array:
        """Convert a Numpy array to an :class:`Array` on this backend, matching shape and type of ``like``."""

    @abstractmethod
    def to_array(self, array: float | bool) -> Array:
        """Convert a Python scalar to an :class:`Array` on this backend."""

    @abstractmethod
    def stack(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        """Stack a sequence of arrays along a new dimension."""

    @abstractmethod
    def reshape(self, array: Array, shape: tuple[int, ...]) -> Array:
        """Reshape ``array`` to ``shape``."""

    @abstractmethod
    def transpose(self, array: Array, axis: tuple[int, ...] | None = None) -> Array:
        """Transpose ``array``; ``None`` reverses the dimensions."""

    @abstractmethod
    def shape(self, array: Array) -> tuple[int, ...]:
        """Return the shape of ``array``."""

    @abstractmethod
    def size(self, array: Array) -> int:
        """Return the total number of elements in ``array``."""

    @abstractmethod
    def ndim(self, array: Array) -> int:
        """Return the number of dimensions of ``array``."""

    @abstractmethod
    def squeeze(self, array: Array, axis: int | tuple[int, ...] | None = None) -> Array:
        """Remove single-dimensional entries from ``array``."""

    @abstractmethod
    def unsqueeze(self, array: Array, axis: int) -> Array:
        """Insert a singleton dimension at ``axis``."""

    @abstractmethod
    def diag(self, array: Array) -> Array:
        """Diagonal: build from a vector or extract from a matrix."""

    @abstractmethod
    def astype(self, array: Array, dtype: type[float | int | bool]) -> float | int | bool:
        """Cast a single-element ``array`` to a Python scalar of ``dtype``."""

    # Linalg --------------------------------------------------------------

    @abstractmethod
    def dot(self, array1: Array, array2: Array) -> Array:
        """Dot product of two arrays."""

    @abstractmethod
    def matmul(self, array1: Array, array2: Array) -> Array:
        """Matrix multiplication of two arrays."""

    @abstractmethod
    def norm(
        self,
        array: Array,
        p: float = 2,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """Compute the norm of ``array``."""

    # Math reductions -----------------------------------------------------

    @abstractmethod
    def sum(self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        """Sum elements of ``array`` along ``axis``."""

    @abstractmethod
    def mean(self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        """Mean of ``array`` along ``axis``."""

    @abstractmethod
    def min(self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        """Minimum of ``array`` along ``axis``."""

    @abstractmethod
    def max(self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        """Maximum of ``array`` along ``axis``."""

    @abstractmethod
    def any(self, array: Array) -> bool:
        """Return True if any element of ``array`` is truthy."""

    @abstractmethod
    def all(self, array: Array) -> bool:
        """Return True if all elements of ``array`` are truthy."""

    # Math elementwise — both operands may be Array or scalar (operator dunders pass
    # either). ``Array | float`` covers both because PEP 484's numeric tower implicitly
    # admits ``int``.

    @abstractmethod
    def add(self, array1: Array | float, array2: Array | float) -> Array:
        """Element-wise addition."""

    @abstractmethod
    def iadd[T: Array](self, array1: T, array2: Array | float) -> T:
        """In-place element-wise addition."""

    @abstractmethod
    def sub(self, array1: Array | float, array2: Array | float) -> Array:
        """Element-wise subtraction."""

    @abstractmethod
    def isub[T: Array](self, array1: T, array2: Array | float) -> T:
        """In-place element-wise subtraction."""

    @abstractmethod
    def mul(self, array1: Array | float, array2: Array | float) -> Array:
        """Element-wise multiplication."""

    @abstractmethod
    def imul[T: Array](self, array1: T, array2: Array | float) -> T:
        """In-place element-wise multiplication."""

    @abstractmethod
    def div(self, array1: Array | float, array2: Array | float) -> Array:
        """Element-wise division."""

    @abstractmethod
    def idiv[T: Array](self, array1: T, array2: Array | float) -> T:
        """In-place element-wise division."""

    @abstractmethod
    def pow(self, array: Array, p: float) -> Array:
        """Raise ``array`` to power ``p``."""

    @abstractmethod
    def negative(self, array: Array) -> Array:
        """Element-wise negation."""

    @abstractmethod
    def absolute(self, array: Array) -> Array:
        """Element-wise absolute value."""

    @abstractmethod
    def sqrt(self, array: Array) -> Array:
        """Element-wise square root."""

    # Comparisons — both operands may be Array or scalar.

    @abstractmethod
    def eq(self, array1: Array | float, array2: Array | float) -> Array:
        """Element-wise equality. Returns an :class:`Array` of bools."""

    @abstractmethod
    def ne(self, array1: Array | float, array2: Array | float) -> Array:
        """Element-wise inequality. Returns an :class:`Array` of bools."""

    @abstractmethod
    def lt(self, array1: Array | float, array2: Array | float) -> Array:
        """Element-wise less-than. Returns an :class:`Array` of bools."""

    @abstractmethod
    def le(self, array1: Array | float, array2: Array | float) -> Array:
        """Element-wise less-than-or-equal. Returns an :class:`Array` of bools."""

    @abstractmethod
    def gt(self, array1: Array | float, array2: Array | float) -> Array:
        """Element-wise greater-than. Returns an :class:`Array` of bools."""

    @abstractmethod
    def ge(self, array1: Array | float, array2: Array | float) -> Array:
        """Element-wise greater-than-or-equal. Returns an :class:`Array` of bools."""

    # Bitwise — operands may be int/bool arrays or scalars. Mirrors Python's ``&``,
    # which dispatches to ``logical_and`` on bool tensors and ``bitwise_and`` on int
    # tensors in every supported framework.

    @abstractmethod
    def bitwise_and(self, array1: Array | float, array2: Array | float) -> Array:
        """Element-wise bitwise/logical AND."""

    # Operators -----------------------------------------------------------

    @abstractmethod
    def sign(self, array: Array) -> Array:
        """Element-wise sign."""

    @abstractmethod
    def maximum(self, array1: Array | float, array2: Array | float) -> Array:
        """Element-wise maximum."""

    @abstractmethod
    def argmax(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        """Index of maximum value along ``axis``."""

    @abstractmethod
    def argmin(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        """Index of minimum value along ``axis``."""

    @abstractmethod
    def set_item(self, array: Array, key: ArrayKey, value: Array) -> None:
        """Set ``array[key] = value``."""

    @abstractmethod
    def get_item(self, array: Array, key: ArrayKey) -> Array:
        """Return ``array[key]``."""

    # RNG -----------------------------------------------------------------

    @abstractmethod
    def set_seed(self, seed: int) -> None:
        """Seed the backend's RNG with ``seed``."""

    @abstractmethod
    def get_rng_state(self) -> dict[str, Any]:
        """Return a snapshot of the backend's RNG state."""

    @abstractmethod
    def set_rng_state(self, state: dict[str, Any]) -> None:
        """Restore an RNG snapshot produced by :meth:`get_rng_state`."""

    @abstractmethod
    def normal(self, mean: float = 0.0, std: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        """Draw normally distributed samples."""

    @abstractmethod
    def uniform(self, low: float = 0.0, high: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        """Draw uniformly distributed samples from ``[low, high)``."""

    @abstractmethod
    def normal_like(self, array: Array, mean: float = 0.0, std: float = 1.0) -> Array:
        """Draw normally distributed samples shaped like ``array``."""

    @abstractmethod
    def uniform_like(self, array: Array, low: float = 0.0, high: float = 1.0) -> Array:
        """Draw uniformly distributed samples shaped like ``array``."""

    @abstractmethod
    def choice(self, array: Array, size: int, replace: bool = True) -> Array:
        """Sample ``size`` elements from ``array``."""
