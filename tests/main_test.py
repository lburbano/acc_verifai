# %%
import pytest

from python_template.main import squared


# %%
@pytest.mark.parametrize(
    ("x", "expected"),
    [
        pytest.param(2, 4, id="positive number"),
        pytest.param(-2, 4, id="negative number"),
    ],
)
def test_squared(x: int, expected: int) -> None:
    """Test the squared function."""
    assert squared(x) == expected


@pytest.mark.skipci(reason="If you didn't have a reason, this would fail.")
def test_squared_but_skip() -> None:
    """Just show how to skip the test."""
    assert squared(3) == 9
