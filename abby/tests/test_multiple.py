from typing import Callable

import numpy as np
import pandas as pd
import pytest

from abby.compare import compare_multiple
from abby.utils import Ratio


@pytest.fixture
def click_df() -> pd.DataFrame:
    data = pd.read_csv("abby/tests/datasets/click_impression.csv")
    return data


@pytest.fixture
def click_df_wrong(click_df):
    return click_df.rename(columns={"variant_name": "group"})


@pytest.fixture
def click_df_mismatch_ratio(click_df: Callable) -> pd.DataFrame:
    return click_df.assign(variant_name=np.repeat(["control", "experiment"], [180, 20]))


class TestCompareMultiple:
    def test_multiple_result(self, click_df: Callable):
        result = compare_multiple(
            click_df, ["control", "experiment"], ["click", Ratio("click", "impression")]
        )
        assert result["click"]["control_mean"] == pytest.approx(0.45, rel=4)
        assert result["click"]["experiment_mean"] == pytest.approx(0.63, rel=4)
        assert result["click"]["control_var"] == pytest.approx(0.35101, rel=4)
        assert result["click"]["experiment_var"] == pytest.approx(0.74050, rel=4)
        assert result["click"]["absolute_difference"] == pytest.approx(0.18, rel=4)
        assert result["click"]["lower_bound"] == pytest.approx(-0.02618, rel=4)
        assert result["click"]["upper_bound"] == pytest.approx(0.38618, rel=4)
        assert result["click"]["p_values"] == pytest.approx(0.08666, rel=4)

        name = "click/impression"
        assert result[name]["control_mean"] == pytest.approx(0.135135, rel=4)
        assert result[name]["experiment_mean"] == pytest.approx(0.162371, rel=4)
        assert result[name]["control_var"] == pytest.approx(0.0295697, rel=4)
        assert result[name]["experiment_var"] == pytest.approx(0.0475355, rel=4)
        assert result[name]["absolute_difference"] == pytest.approx(0.027236, rel=4)
        assert result[name]["lower_bound"] == pytest.approx(-0.027189, rel=4)
        assert result[name]["upper_bound"] == pytest.approx(0.081661, rel=4)
        assert result[name]["p_values"] == pytest.approx(0.326668, rel=4)

    def test_multiple_wrong_variant_column_name(self, click_df_wrong: Callable):
        with pytest.raises(AssertionError):
            compare_multiple(
                click_df_wrong,
                ["control", "experiment"],
                ["click", Ratio("click", "impression")],
            )

    def test_multiple_raise_warning_mismatch_ratio(
        self, click_df_mismatch_ratio: Callable
    ):
        with pytest.warns(UserWarning):
            compare_multiple(
                click_df_mismatch_ratio,
                ["control", "experiment"],
                ["click", Ratio("click", "impression")],
            )
