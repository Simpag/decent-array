"""Tests for the module-level functions in :mod:`decent_array.interoperability.iop.functions`."""

from __future__ import annotations

import numpy as np
import pytest

import decent_array as da
from decent_array.array import Array
from decent_array.interoperability.backend_manager import reset_backends
from decent_array.interoperability.iop import functions as iop_functions
from decent_array.types import SupportedDevices


def _np(arr: Array) -> np.ndarray:
    return da.to_numpy(arr)


# Array creation ---------------------------------------------------------


def test_zeros(backend: tuple) -> None:
    arr = da.zeros((2, 3))
    np.testing.assert_allclose(_np(arr), np.zeros((2, 3)))


def test_zeros_like(backend: tuple) -> None:
    src = da.from_numpy(np.ones((2, 3)))
    arr = da.zeros_like(src)
    np.testing.assert_allclose(_np(arr), np.zeros((2, 3)))


def test_ones(backend: tuple) -> None:
    arr = da.ones((2, 3))
    np.testing.assert_allclose(_np(arr), np.ones((2, 3)))


def test_ones_like(backend: tuple) -> None:
    src = da.from_numpy(np.zeros((2, 3)))
    arr = da.ones_like(src)
    np.testing.assert_allclose(_np(arr), np.ones((2, 3)))


def test_eye(backend: tuple) -> None:
    arr = da.eye(3)
    np.testing.assert_allclose(_np(arr), np.eye(3))


def test_eye_like(backend: tuple) -> None:
    src = da.from_numpy(np.zeros((4, 4)))
    arr = da.eye_like(src)
    np.testing.assert_allclose(_np(arr), np.eye(4))


def test_device_to_native(backend: tuple) -> None:
    framework, device = backend
    native = da.device_to_native(device)
    # Just verify the call succeeds and returns a non-None value (varies per backend).
    assert native is not None


def test_device_of(backend: tuple) -> None:
    _framework, device = backend
    arr = da.zeros((3,))
    assert da.device_of(arr) == device


# Array manipulation -----------------------------------------------------


def test_copy_independent(backend: tuple) -> None:
    src = da.from_numpy(np.array([1.0, 2.0, 3.0]))
    dst = da.copy(src)
    np.testing.assert_allclose(_np(dst), [1.0, 2.0, 3.0])
    # Mutating the copy shouldn't affect the original.
    dst[0] = 99.0
    np.testing.assert_allclose(_np(src), [1.0, 2.0, 3.0])


def test_to_numpy(backend: tuple) -> None:
    arr = da.from_numpy(np.array([1.0, 2.0, 3.0]))
    out = da.to_numpy(arr)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


def test_from_numpy_roundtrip(backend: tuple) -> None:
    raw = np.array([[1.0, 2.0], [3.0, 4.0]])
    arr = da.from_numpy(raw)
    np.testing.assert_allclose(_np(arr), raw)


def test_to_array_from_scalar(backend: tuple) -> None:
    arr = da.to_array(2.5)
    np.testing.assert_allclose(_np(arr), 2.5)


def test_stack(backend: tuple) -> None:
    a = da.from_numpy(np.array([1.0, 2.0]))
    b = da.from_numpy(np.array([3.0, 4.0]))
    arr = da.stack([a, b])
    np.testing.assert_allclose(_np(arr), [[1.0, 2.0], [3.0, 4.0]])


def test_stack_dim(backend: tuple) -> None:
    a = da.from_numpy(np.array([1.0, 2.0]))
    b = da.from_numpy(np.array([3.0, 4.0]))
    arr = da.stack([a, b], dim=1)
    np.testing.assert_allclose(_np(arr), [[1.0, 3.0], [2.0, 4.0]])


def test_stack_empty_raises(backend: tuple) -> None:
    with pytest.raises(ValueError, match=r"empty sequence"):
        da.stack([])


def test_reshape(backend: tuple) -> None:
    arr = da.from_numpy(np.arange(6, dtype=np.float32))
    out = da.reshape(arr, (2, 3))
    np.testing.assert_allclose(_np(out), np.arange(6, dtype=np.float32).reshape(2, 3))


def test_transpose_default(backend: tuple) -> None:
    arr = da.from_numpy(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    np.testing.assert_allclose(_np(da.transpose(arr)), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])


def test_transpose_explicit_dim(backend: tuple) -> None:
    arr = da.from_numpy(np.zeros((2, 3, 4), dtype=np.float32))
    out = da.transpose(arr, dim=(1, 0, 2))
    assert da.shape(out) == (3, 2, 4)


def test_shape_function(backend: tuple) -> None:
    arr = da.from_numpy(np.zeros((2, 3, 4)))
    assert da.shape(arr) == (2, 3, 4)


def test_size_function(backend: tuple) -> None:
    arr = da.from_numpy(np.zeros((2, 3, 4)))
    assert da.size(arr) == 24


def test_ndim_function(backend: tuple) -> None:
    arr = da.from_numpy(np.zeros((2, 3, 4)))
    assert da.ndim(arr) == 3


def test_squeeze_default(backend: tuple) -> None:
    arr = da.from_numpy(np.zeros((1, 3, 1)))
    assert da.shape(da.squeeze(arr)) == (3,)


def test_squeeze_specific_dim(backend: tuple) -> None:
    arr = da.from_numpy(np.zeros((1, 3, 1)))
    assert da.shape(da.squeeze(arr, dim=0)) == (3, 1)


def test_unsqueeze(backend: tuple) -> None:
    arr = da.from_numpy(np.zeros((3,)))
    assert da.shape(da.unsqueeze(arr, dim=0)) == (1, 3)
    assert da.shape(da.unsqueeze(arr, dim=1)) == (3, 1)


def test_diag_from_vector(backend: tuple) -> None:
    arr = da.from_numpy(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(_np(da.diag(arr)), np.diag([1.0, 2.0, 3.0]))


def test_diag_from_matrix(backend: tuple) -> None:
    arr = da.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(_np(da.diag(arr)), [1.0, 4.0])


def test_astype_to_float(backend: tuple) -> None:
    arr = da.to_array(3.0)
    out = da.astype(arr, float)
    assert isinstance(out, float)
    assert out == pytest.approx(3.0)


def test_astype_to_int(backend: tuple) -> None:
    arr = da.to_array(3.0)
    out = da.astype(arr, int)
    assert isinstance(out, int)
    assert out == 3


def test_astype_to_bool(backend: tuple) -> None:
    arr = da.to_array(1.0)
    out = da.astype(arr, bool)
    assert isinstance(out, bool)
    assert out is True


# Linalg -----------------------------------------------------------------


def test_dot(backend: tuple) -> None:
    a = da.from_numpy(np.array([1.0, 2.0, 3.0]))
    b = da.from_numpy(np.array([4.0, 5.0, 6.0]))
    np.testing.assert_allclose(_np(da.dot(a, b)), 32.0)


def test_matmul(backend: tuple) -> None:
    a = da.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = da.from_numpy(np.array([[5.0, 6.0], [7.0, 8.0]]))
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    np.testing.assert_allclose(_np(da.matmul(a, b)), expected)


def test_norm_default_l2(backend: tuple) -> None:
    arr = da.from_numpy(np.array([3.0, 4.0]))
    np.testing.assert_allclose(_np(da.norm(arr)), 5.0)


def test_norm_p1(backend: tuple) -> None:
    arr = da.from_numpy(np.array([3.0, -4.0]))
    np.testing.assert_allclose(_np(da.norm(arr, p=1)), 7.0)


def test_norm_dim_keepdims(backend: tuple) -> None:
    arr = da.from_numpy(np.array([[3.0, 4.0], [6.0, 8.0]]))
    out = da.norm(arr, p=2, dim=1, keepdims=True)
    assert da.shape(out) == (2, 1)
    np.testing.assert_allclose(_np(out).reshape(-1), [5.0, 10.0])


# Math reductions --------------------------------------------------------


def test_sum_all(backend: tuple) -> None:
    arr = da.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(_np(da.sum(arr)), 10.0)


def test_sum_dim(backend: tuple) -> None:
    arr = da.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(_np(da.sum(arr, dim=0)), [4.0, 6.0])


def test_sum_dim_keepdims(backend: tuple) -> None:
    arr = da.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
    out = da.sum(arr, dim=0, keepdims=True)
    assert da.shape(out) == (1, 2)


def test_mean_all(backend: tuple) -> None:
    arr = da.from_numpy(np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(_np(da.mean(arr)), 2.5)


def test_mean_dim(backend: tuple) -> None:
    arr = da.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(_np(da.mean(arr, dim=1)), [1.5, 3.5])


def test_min_all(backend: tuple) -> None:
    arr = da.from_numpy(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))
    np.testing.assert_allclose(_np(da.min(arr)), 1.0)


def test_min_dim(backend: tuple) -> None:
    arr = da.from_numpy(np.array([[1.0, 5.0], [3.0, 2.0]]))
    np.testing.assert_allclose(_np(da.min(arr, dim=0)), [1.0, 2.0])


def test_max_all(backend: tuple) -> None:
    arr = da.from_numpy(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))
    np.testing.assert_allclose(_np(da.max(arr)), 5.0)


def test_max_dim(backend: tuple) -> None:
    arr = da.from_numpy(np.array([[1.0, 5.0], [3.0, 2.0]]))
    np.testing.assert_allclose(_np(da.max(arr, dim=0)), [3.0, 5.0])


def test_any_true(backend: tuple) -> None:
    arr = da.from_numpy(np.array([0.0, 1.0, 0.0]))
    assert da.any(arr) is True


def test_any_false(backend: tuple) -> None:
    arr = da.from_numpy(np.array([0.0, 0.0, 0.0]))
    assert da.any(arr) is False


def test_all_true(backend: tuple) -> None:
    arr = da.from_numpy(np.array([1.0, 2.0, 3.0]))
    assert da.all(arr) is True


def test_all_false(backend: tuple) -> None:
    arr = da.from_numpy(np.array([1.0, 0.0, 3.0]))
    assert da.all(arr) is False


# Math elementwise -------------------------------------------------------


def test_add_two_arrays(backend: tuple) -> None:
    a = da.from_numpy(np.array([1.0, 2.0]))
    b = da.from_numpy(np.array([3.0, 4.0]))
    np.testing.assert_allclose(_np(da.add(a, b)), [4.0, 6.0])


def test_add_array_and_scalar(backend: tuple) -> None:
    a = da.from_numpy(np.array([1.0, 2.0]))
    np.testing.assert_allclose(_np(da.add(a, 10.0)), [11.0, 12.0])


def test_sub(backend: tuple) -> None:
    a = da.from_numpy(np.array([5.0, 6.0]))
    b = da.from_numpy(np.array([1.0, 2.0]))
    np.testing.assert_allclose(_np(da.sub(a, b)), [4.0, 4.0])


def test_mul(backend: tuple) -> None:
    a = da.from_numpy(np.array([2.0, 3.0]))
    b = da.from_numpy(np.array([4.0, 5.0]))
    np.testing.assert_allclose(_np(da.mul(a, b)), [8.0, 15.0])


def test_div(backend: tuple) -> None:
    a = da.from_numpy(np.array([8.0, 10.0]))
    b = da.from_numpy(np.array([2.0, 5.0]))
    np.testing.assert_allclose(_np(da.div(a, b)), [4.0, 2.0])


def test_iadd_func(backend: tuple) -> None:
    a = da.from_numpy(np.array([1.0, 2.0]))
    out = da.iadd(a, 10.0)
    # Returned wrapper is the same instance.
    assert out is a
    np.testing.assert_allclose(_np(a), [11.0, 12.0])


def test_isub_func(backend: tuple) -> None:
    a = da.from_numpy(np.array([5.0, 6.0]))
    out = da.isub(a, 1.0)
    assert out is a
    np.testing.assert_allclose(_np(a), [4.0, 5.0])


def test_imul_func(backend: tuple) -> None:
    a = da.from_numpy(np.array([2.0, 3.0]))
    out = da.imul(a, 4.0)
    assert out is a
    np.testing.assert_allclose(_np(a), [8.0, 12.0])


def test_idiv_func(backend: tuple) -> None:
    a = da.from_numpy(np.array([8.0, 12.0]))
    out = da.idiv(a, 4.0)
    assert out is a
    np.testing.assert_allclose(_np(a), [2.0, 3.0])


def test_pow_function(backend: tuple) -> None:
    arr = da.from_numpy(np.array([2.0, 3.0, 4.0]))
    np.testing.assert_allclose(_np(da.pow(arr, 2)), [4.0, 9.0, 16.0])


def test_negative(backend: tuple) -> None:
    arr = da.from_numpy(np.array([1.0, -2.0, 3.0]))
    np.testing.assert_allclose(_np(da.negative(arr)), [-1.0, 2.0, -3.0])


def test_absolute(backend: tuple) -> None:
    arr = da.from_numpy(np.array([1.0, -2.0, 3.0]))
    np.testing.assert_allclose(_np(da.absolute(arr)), [1.0, 2.0, 3.0])


def test_sqrt(backend: tuple) -> None:
    arr = da.from_numpy(np.array([1.0, 4.0, 9.0]))
    np.testing.assert_allclose(_np(da.sqrt(arr)), [1.0, 2.0, 3.0])


# Operators --------------------------------------------------------------


def test_sign(backend: tuple) -> None:
    arr = da.from_numpy(np.array([-2.0, 0.0, 3.0]))
    np.testing.assert_allclose(_np(da.sign(arr)), [-1.0, 0.0, 1.0])


def test_maximum_arrays(backend: tuple) -> None:
    a = da.from_numpy(np.array([1.0, 5.0, 3.0]))
    b = da.from_numpy(np.array([4.0, 2.0, 6.0]))
    np.testing.assert_allclose(_np(da.maximum(a, b)), [4.0, 5.0, 6.0])


def test_maximum_array_and_scalar(backend: tuple) -> None:
    a = da.from_numpy(np.array([1.0, 5.0, 3.0]))
    np.testing.assert_allclose(_np(da.maximum(a, 4.0)), [4.0, 5.0, 4.0])


def test_argmax_default(backend: tuple) -> None:
    arr = da.from_numpy(np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]))
    np.testing.assert_allclose(_np(da.argmax(arr)), 5)


def test_argmax_dim(backend: tuple) -> None:
    arr = da.from_numpy(np.array([[1.0, 5.0, 2.0], [4.0, 0.0, 3.0]]))
    np.testing.assert_allclose(_np(da.argmax(arr, dim=1)), [1, 0])


def test_argmin_default(backend: tuple) -> None:
    arr = da.from_numpy(np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]))
    # First occurrence of minimum is index 1.
    np.testing.assert_allclose(_np(da.argmin(arr)), 1)


def test_argmin_dim(backend: tuple) -> None:
    arr = da.from_numpy(np.array([[1.0, 5.0, 2.0], [4.0, 0.0, 3.0]]))
    np.testing.assert_allclose(_np(da.argmin(arr, dim=1)), [0, 1])


def test_argmax_keepdims(backend: tuple) -> None:
    arr = da.from_numpy(np.array([[1.0, 5.0, 2.0], [4.0, 0.0, 3.0]]))
    out = da.argmax(arr, dim=1, keepdims=True)
    assert da.shape(out) == (2, 1)


def test_set_item_function(backend: tuple) -> None:
    arr = da.from_numpy(np.array([1.0, 2.0, 3.0]))
    da.set_item(arr, 0, da.from_numpy(np.array(99.0)))
    np.testing.assert_allclose(_np(arr), [99.0, 2.0, 3.0])


def test_get_item_function(backend: tuple) -> None:
    arr = da.from_numpy(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(_np(da.get_item(arr, 1)), 2.0)


# No-backend errors ------------------------------------------------------


def test_function_raises_when_no_backend() -> None:
    reset_backends()
    with pytest.raises(RuntimeError, match=r"No backend active"):
        iop_functions.zeros((3,))


def test_to_array_round_trip_with_bool(backend: tuple) -> None:
    arr = da.to_array(True)
    out = da.astype(arr, bool)
    assert out is True
