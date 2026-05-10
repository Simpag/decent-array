"""Tests for :mod:`decent_array.interoperability.decorators`.

Note: this file deliberately omits ``from __future__ import annotations``. The
:func:`autodecorate_cost_method` decorator inspects ``__annotations__["return"] is Array``
on the superclass method, and PEP 563-style stringified annotations would break that
identity check.
"""

from typing import Any

import numpy as np

import decent_array.interoperability as iop
from decent_array import Array
from decent_array.interoperability._decorators import autodecorate_cost_method


class _Base:
    def returns_array(self, x: Array) -> Array:
        raise NotImplementedError

    def returns_float(self, x: Array) -> float:
        raise NotImplementedError

    def takes_kwarg(self, x: Array, *, scale: Array) -> Array:
        raise NotImplementedError

    def passes_through_non_array(self, x: Array, n: int) -> Array:
        raise NotImplementedError


def test_unwraps_array_args(backend: tuple) -> None:
    seen: list[Any] = []

    class Impl(_Base):
        @autodecorate_cost_method(_Base.returns_array)
        def returns_array(self, x: Any) -> Any:
            seen.append(x)
            return x * 2

    arr = iop.from_numpy(np.array([1.0, 2.0, 3.0]))
    result = Impl().returns_array(arr)
    # The decorator should have unwrapped the Array to its underlying value.
    assert not isinstance(seen[0], Array)
    # Annotated `-> Array`, so the return is re-wrapped.
    assert isinstance(result, Array)
    np.testing.assert_allclose(iop.to_numpy(result), [2.0, 4.0, 6.0])


def test_does_not_rewrap_when_return_not_array(backend: tuple) -> None:
    class Impl(_Base):
        @autodecorate_cost_method(_Base.returns_float)
        def returns_float(self, x: Any) -> float:
            # Use a fixed value so the test doesn't depend on framework-specific tensor
            # methods (e.g. TF eager tensors don't expose ``.sum()`` without enabling
            # the numpy compatibility shim).
            assert not isinstance(x, Array)
            return 42.0

    arr = iop.from_numpy(np.array([1.0, 2.0, 3.0]))
    result = Impl().returns_float(arr)
    assert isinstance(result, float)
    assert result == 42.0


def test_unwraps_kwargs(backend: tuple) -> None:
    class Impl(_Base):
        @autodecorate_cost_method(_Base.takes_kwarg)
        def takes_kwarg(self, x: Any, *, scale: Any) -> Any:
            assert not isinstance(x, Array)
            assert not isinstance(scale, Array)
            return x * scale

    arr = iop.from_numpy(np.array([1.0, 2.0]))
    scale = iop.from_numpy(np.array([10.0, 100.0]))
    result = Impl().takes_kwarg(arr, scale=scale)
    assert isinstance(result, Array)
    np.testing.assert_allclose(iop.to_numpy(result), [10.0, 200.0])


def test_passes_non_array_args_unchanged(backend: tuple) -> None:
    class Impl(_Base):
        @autodecorate_cost_method(_Base.passes_through_non_array)
        def passes_through_non_array(self, x: Any, n: int) -> Any:
            assert isinstance(n, int)
            return x * n

    arr = iop.from_numpy(np.array([1.0, 2.0, 3.0]))
    result = Impl().passes_through_non_array(arr, 4)
    assert isinstance(result, Array)
    np.testing.assert_allclose(iop.to_numpy(result), [4.0, 8.0, 12.0])


def test_does_not_double_wrap_when_impl_returns_array(backend: tuple) -> None:
    """If the decorated impl already returns an :class:`Array`, the wrapper should not double-wrap."""

    class Impl(_Base):
        @autodecorate_cost_method(_Base.returns_array)
        def returns_array(self, x: Any) -> Array:
            return Array(x * 3)

    arr = iop.from_numpy(np.array([1.0, 2.0]))
    result = Impl().returns_array(arr)
    assert isinstance(result, Array)
    # If the wrapper double-wrapped, ``result.value`` would itself be an Array.
    assert not isinstance(result.value, Array)
    np.testing.assert_allclose(iop.to_numpy(result), [3.0, 6.0])
