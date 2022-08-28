from typing import Callable

import pandas as pd
import pytest

from abby.compare import compare_bootstrap_delta


@pytest.fixture
def click_df() -> pd.DataFrame:
    data = pd.read_csv("abby/tests/datasets/click_impression.csv")
    return data


@pytest.fixture
def click_df_wrong(click_df):
    return click_df.rename(columns={"variant_name": "group"})


class TestCompareBootstrap:
    def test_bootstrap_result(self, click_df: Callable):
        result = compare_bootstrap_delta(
            click_df, ["control", "experiment"], "click", "impression"
        )
        assert result["control_mean"] == pytest.approx(0.135135, rel=2)
        assert result["experiment_mean"] == pytest.approx(0.162371, rel=2)
        assert result["control_var"] == pytest.approx(0.000296, rel=2)
        assert result["experiment_var"] == pytest.approx(0.000475, rel=2)
        assert result["absolute_difference"] == pytest.approx(0.027236, rel=2)
        assert result["lower_bound"] == pytest.approx(-0.027189, rel=2)
        assert result["upper_bound"] == pytest.approx(0.081661, rel=2)
        assert result["p_values"] == pytest.approx(0.326668, rel=2)

    def test_bootstrap_wrong_variant_column_name(self, click_df_wrong: Callable):
        with pytest.raises(AssertionError):
            compare_bootstrap_delta(
                click_df_wrong, ["control", "experiment"], "click", "impression"
            )
