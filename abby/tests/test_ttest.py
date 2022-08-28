from typing import Callable

import pandas as pd
import pytest

from abby.compare import compare_ttest


@pytest.fixture
def sleeps() -> pd.DataFrame:
    data = pd.read_csv("abby/tests/datasets/sleeps.csv").assign(
        variant_name=lambda df: df["group"].map({1: "control", 2: "experiment"})
    )
    return data


@pytest.fixture
def sleeps_wrong(sleeps):
    return sleeps.rename(columns={"variant_name": "group"})


class TestCompareTtest:
    def test_ttest_result(self, sleeps: Callable):
        result = compare_ttest(sleeps, ["control", "experiment"], "extra")
        assert result["control_mean"] == pytest.approx(0.75, rel=4)
        assert result["experiment_mean"] == pytest.approx(2.33, rel=4)
        assert result["control_var"] == pytest.approx(3.200556, rel=4)
        assert result["experiment_var"] == pytest.approx(4.009000, rel=4)
        assert result["absolute_difference"] == pytest.approx(1.5800, rel=4)
        assert result["lower_bound"] == pytest.approx(-0.2054832, rel=4)
        assert result["upper_bound"] == pytest.approx(3.3654832, rel=4)
        assert result["p_values"] == pytest.approx(0.07939, rel=4)

    def test_ttest_wrong_variant_column_name(self, sleeps_wrong: Callable):
        with pytest.raises(AssertionError):
            compare_ttest(sleeps_wrong, ["control", "experiment"], "extra")
