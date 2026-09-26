import numpy as np
import pytest

from sn_hunter.imaging import stretch_image


@pytest.mark.parametrize("mode", ["linear", "sqrt", "log", "asinh", "power"])
def test_stretch_is_finite_and_bounded(mode):
    data = np.array([[np.nan, -3.0, 0.0], [1.0, 5.0, 100.0]])
    result = stretch_image(data, mode, 0, 100, 2)
    assert np.isfinite(result).all()
    assert result.min() >= 0
    assert result.max() <= 1


def test_unknown_stretch_rejected():
    with pytest.raises(ValueError):
        stretch_image(np.arange(4).reshape(2, 2), "banana")

