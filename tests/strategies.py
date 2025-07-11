from math import isclose
from math import pi as π

import hypothesis.extra.numpy as hn
import numpy as np
import numpy.typing as npt
from hypothesis import strategies as st

from stellar.gaussian import GaussianParameters
from stellar.states import StateFockBasis


@st.composite
def complex_arrays_st(draw, min_size=1, max_size=10):
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
def unit_norm_complex_arrays_st(draw, min_size=1, max_size=10):
    # avoid repetition somehow?
    # arr = complex_arrays(draw, min_size=min_size, max_size=max)

    # Draw the array size
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    # or sv_shape_st = hn.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=10)

    # Draw a random complex array
    arr = draw(complex_arrays_st(size, size))
    # Normalize to unit norm
    new_norm = np.sqrt(np.sum(np.abs(arr) ** 2))
    arr = arr / new_norm
    return arr


@st.composite
def tuple_StateFockBasis_st(draw, min_size=1, max_size=10) -> tuple[StateFockBasis, StateFockBasis]:
    """generate a pair of StateFockBasis objects with same size"""
    # avoid repetition somehow?
    # arr = complex_arrays(draw, min_size=min_size, max_size=max)

    # Draw the array size
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    # or sv_shape_st = hn.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=10)

    # Draw a random complex array of fixed size
    arr1 = draw(unit_norm_complex_arrays_st(size, size))

    arr2 = draw(unit_norm_complex_arrays_st(size, size))

    return StateFockBasis(arr1), StateFockBasis(arr2)


@st.composite
def tuple_StateFockBasis_mat_left_st(
    draw, min_size=1, max_size=10
) -> tuple[StateFockBasis, npt.NDArray[np.complex128]]:  # no support for annotating number of dimensions
    """generate a pair of (StateFockBasis, mat) with matchin left dimensions"""
    # avoid repetition somehow?
    # arr = complex_arrays(draw, min_size=min_size, max_size=max)

    # Draw the array size
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    size2 = draw(st.integers(min_value=min_size, max_value=max_size))
    # or sv_shape_st = hn.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=10)

    # Draw a random complex array
    arr = draw(unit_norm_complex_arrays_st(size, size))

    mat = draw(
        hn.arrays(
            dtype=np.complex128,
            shape=(size, size2),
            elements=st.complex_numbers(allow_nan=False, allow_infinity=False, max_magnitude=1e5),
        )
    )

    return StateFockBasis(arr), mat


@st.composite
def tuple_ints_fock_cutoff_st(draw, min_size=1, max_size=20) -> tuple[int, int]:
    # Draw the array size
    cutoff = draw(st.integers(min_value=min_size, max_value=max_size))
    n = draw(st.integers(min_value=0, max_value=cutoff - 1))

    return n, cutoff


@st.composite
def gaussian_parameters_st(draw) -> GaussianParameters:  # tuple[float, float, float, float]:
    x = draw(st.floats(min_value=-10, max_value=10))
    y = draw(st.floats(min_value=-10, max_value=10))
    r = draw(st.floats(min_value=0, max_value=20))
    theta = draw(st.floats(min_value=0, max_value=2 * π))
    # apparently cannot cast to custum GaussianParameter
    return GaussianParameters(x=x, y=y, r=r, theta=theta)
