import logging
from math import isclose

import hypothesis.extra.numpy as hn
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from stellar.states import StateFockBasis, Statevector

logger = logging.getLogger(__name__)

logger.info("Starting state tests")

# use composite for normalisation
# @st.composite
# def sums_to_one(draw):
#     l = draw(st.lists(st.floats(0, 1)))
#     return [f / sum(l) for f in l]
# need to avoid the all zero vector

# put in other file to share


@st.composite
def complex_arrays(draw, min_size=1, max_size=10):
    """custom strategy to draw 1-dim complex arrays while avoiding the all-zero vector."""
    # Draw the array size
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    # or sv_shape_st = hn.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=10)

    # Draw a random complex array
    arr = draw(
        hn.arrays(
            dtype=np.complex128,
            shape=(size,),
            elements=st.complex_numbers(allow_nan=False, allow_infinity=False, max_magnitude=1e5),
        )
    )
    # Avoid zero vector
    if isclose(np.sqrt(np.sum(np.abs(arr) ** 2)), 0):
        arr[0] = 1.0 + 2j
    return arr


# NOTE to refactor to avoid code duplication. But maybe it's useless.
@st.composite
def unit_norm_complex_arrays(draw, min_size=1, max_size=10):
    # avoid repetition somehow?
    # arr = complex_arrays(draw, min_size=min_size, max_size=max)

    # Draw the array size
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    # or sv_shape_st = hn.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=10)

    # Draw a random complex array
    arr = draw(
        hn.arrays(
            dtype=np.complex128,
            shape=(size,),
            elements=st.complex_numbers(allow_nan=False, allow_infinity=False, max_magnitude=1e5),
        )
    )

    # Avoid zero vector
    norm = np.sqrt(np.sum(np.abs(arr) ** 2))
    if isclose(norm, 0):
        arr[0] = 1.0 + 2j

    # Normalize to unit norm
    new_norm = np.sqrt(np.sum(np.abs(arr) ** 2))
    arr = arr / new_norm
    return arr


@given(complex_arrays())
def test_init_success(data: Statevector) -> None:
    state = StateFockBasis(data)
    assert state.dim > 0


# failures?


@given(unit_norm_complex_arrays())
def test_norm(data: Statevector) -> None:
    state = StateFockBasis(data)
    print(state)
    assert isclose(state.get_norm(), 1)


@given(complex_arrays())
def test_normalize(data: Statevector) -> None:
    state = StateFockBasis(data)
    state.normalize()
    assert isclose(state.get_norm(), 1, abs_tol=1e-6)
