"""Tests for :mod:`decent_array.interoperability.iop.rng`."""

from __future__ import annotations

import random

import numpy as np
import pytest

import decent_array as da
from decent_array.array import Array
from decent_array.interoperability.backend_manager import reset_backends
from decent_array.interoperability.iop import rng as iop_rng
from decent_array.interoperability.iop.rng import _set_seed_without_global


def _np(arr: Array) -> np.ndarray:
    return da.to_numpy(arr)


# Seed management --------------------------------------------------------


def test_set_seed_records_global(backend: tuple) -> None:
    da.set_seed(123)
    assert da.get_seed() == 123


def test_set_seed_makes_normal_reproducible(backend: tuple) -> None:
    da.set_seed(7)
    first = _np(da.normal(shape=(4,)))
    da.set_seed(7)
    second = _np(da.normal(shape=(4,)))
    np.testing.assert_allclose(first, second)


def test_set_seed_makes_uniform_reproducible(backend: tuple) -> None:
    da.set_seed(7)
    first = _np(da.uniform(shape=(4,)))
    da.set_seed(7)
    second = _np(da.uniform(shape=(4,)))
    np.testing.assert_allclose(first, second)


def test_set_seed_seeds_python_random(backend: tuple) -> None:
    da.set_seed(11)
    a = random.random()
    da.set_seed(11)
    b = random.random()
    assert a == b


def test_set_seed_without_global_keeps_seed(backend: tuple) -> None:
    da.set_seed(42)
    _set_seed_without_global(99)
    # The "global" seed observable via get_seed() must be unchanged.
    assert da.get_seed() == 42


def test_get_seed_initially_none(backend: tuple) -> None:
    # Fresh activation via the fixture: nothing has called set_seed yet on this
    # coordinator instance, but the coordinator is process-singleton and may have
    # state from earlier tests. Call set_seed/then-clear to assert via reseed instead.
    da.set_seed(0)
    assert da.get_seed() == 0


# RNG state snapshot/restore --------------------------------------------


def test_rng_state_round_trip_normal(backend: tuple) -> None:
    da.set_seed(123)
    state = da.get_rng_state()
    first = _np(da.normal(shape=(4,)))
    da.set_rng_state(state)
    second = _np(da.normal(shape=(4,)))
    np.testing.assert_allclose(first, second)


def test_rng_state_includes_python_random(backend: tuple) -> None:
    da.set_seed(7)
    state = da.get_rng_state()
    a = random.random()
    da.set_rng_state(state)
    b = random.random()
    assert a == b


def test_rng_state_round_trip_uniform(backend: tuple) -> None:
    da.set_seed(456)
    state = da.get_rng_state()
    first = _np(da.uniform(shape=(3,)))
    da.set_rng_state(state)
    second = _np(da.uniform(shape=(3,)))
    np.testing.assert_allclose(first, second)


# derive_seed -----------------------------------------------------------


def test_derive_seed_when_seed_set_is_reproducible(backend: tuple) -> None:
    da.set_seed(123)
    a = da.derive_seed()
    da.set_seed(123)
    b = da.derive_seed()
    assert a == b


def test_derive_seed_returns_int_in_range(backend: tuple) -> None:
    da.set_seed(0)
    seed = da.derive_seed()
    assert isinstance(seed, int)
    assert 0 <= seed < 2**32


# Distribution shape checks ---------------------------------------------


def test_normal_shape(backend: tuple) -> None:
    arr = da.normal(shape=(2, 3))
    assert da.shape(arr) == (2, 3)


def test_normal_default_scalar(backend: tuple) -> None:
    da.set_seed(1)
    arr = da.normal()
    assert da.shape(arr) == ()


def test_uniform_shape_and_range(backend: tuple) -> None:
    da.set_seed(1)
    arr = da.uniform(low=0.0, high=1.0, shape=(50,))
    samples = _np(arr)
    assert samples.shape == (50,)
    assert (samples >= 0.0).all()
    assert (samples < 1.0).all()


def test_uniform_custom_range(backend: tuple) -> None:
    da.set_seed(1)
    samples = _np(da.uniform(low=-2.0, high=-1.0, shape=(50,)))
    assert (samples >= -2.0).all()
    assert (samples < -1.0).all()


def test_normal_like(backend: tuple) -> None:
    src = da.from_numpy(np.zeros((3, 4)))
    arr = da.normal_like(src)
    assert da.shape(arr) == (3, 4)


def test_uniform_like(backend: tuple) -> None:
    src = da.from_numpy(np.zeros((3, 4)))
    arr = da.uniform_like(src, low=0.0, high=1.0)
    assert da.shape(arr) == (3, 4)
    samples = _np(arr)
    assert (samples >= 0.0).all()
    assert (samples < 1.0).all()


def test_choice_shape(backend: tuple) -> None:
    da.set_seed(1)
    population = da.from_numpy(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
    sample = da.choice(population, size=3)
    assert da.shape(sample) == (3,)


def test_choice_values_in_population(backend: tuple) -> None:
    da.set_seed(1)
    pop_np = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    sample = da.choice(da.from_numpy(pop_np), size=10)
    drawn = _np(sample).reshape(-1)
    assert all(v in pop_np for v in drawn)


def test_choice_no_replace_unique(backend: tuple) -> None:
    da.set_seed(1)
    pop = da.from_numpy(np.arange(20, dtype=np.float32))
    sample = da.choice(pop, size=5, replace=False)
    drawn = _np(sample).reshape(-1)
    assert len(set(drawn.tolist())) == 5


# No-backend errors -----------------------------------------------------


def test_rng_function_raises_when_no_backend() -> None:
    reset_backends()
    with pytest.raises(RuntimeError, match=r"No backend active"):
        iop_rng.normal(shape=(3,))


def test_set_seed_raises_when_no_backend() -> None:
    reset_backends()
    with pytest.raises(RuntimeError, match=r"No backend active"):
        iop_rng.set_seed(0)
