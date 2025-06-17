from hypothesis import strategies as st
import hypothesis.extra.numpy as hn
import numpy as np
from math import isclose


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
