import pytest

from stellar.params import Method, OptimisationParameters


def test_init_fail_fock() -> None:
    with pytest.raises(ValueError):
        OptimisationParameters(method=Method.fock)

def test_init_warn_gaussian() -> None:
    with pytest.warns(match="`target_cutoff` will be ignored using the `gaussian` method."):
        OptimisationParameters(method=Method.gaussian, target_cutoff=3)
