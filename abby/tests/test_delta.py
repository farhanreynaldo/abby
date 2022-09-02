from typing import Callable

import numpy as np
import pandas as pd
import pytest

from abby.compare import compare_delta


@pytest.fixture
def click_df() -> pd.DataFrame:
    data = pd.read_csv("abby/tests/datasets/click_impression.csv")
    return data


@pytest.fixture
def click_df_wrong(click_df: Callable) -> pd.DataFrame:
    return click_df.rename(columns={"variant_name": "group"})


@pytest.fixture
def click_df_mismatch_ratio(click_df: Callable) -> pd.DataFrame:
    return click_df.assign(variant_name=np.repeat(["control", "experiment"], [180, 20]))


class TestCompareDelta:
    def test_delta_result(self, click_df: Callable):
        result = compare_delta(
            click_df, ["control", "experiment"], "click", "impression"
        )
        assert result["control_mean"] == pytest.approx(0.135135, rel=4)
        assert result["experiment_mean"] == pytest.approx(0.162371, rel=4)
        assert result["control_var"] == pytest.approx(0.000296, rel=4)
        assert result["experiment_var"] == pytest.approx(0.000475, rel=4)
        assert result["absolute_difference"] == pytest.approx(0.027236, rel=4)
        assert result["lower_bound"] == pytest.approx(-0.027189, rel=4)
        assert result["upper_bound"] == pytest.approx(0.081661, rel=4)
        assert result["p_values"] == pytest.approx(0.326668, rel=4)

    def test_delta_wrong_variant_column_name(self, click_df_wrong: Callable):
        with pytest.raises(AssertionError):
            compare_delta(
                click_df_wrong, ["control", "experiment"], "click", "impression"
            )

    def test_delta_raise_warning_mismatch_ratio(
        self, click_df_mismatch_ratio: Callable
    ):
        with pytest.warns(UserWarning):
            compare_delta(
                click_df_mismatch_ratio,
                ["control", "experiment"],
                "click",
                "impression",
            )
