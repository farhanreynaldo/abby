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
def click_multi_df() -> pd.DataFrame:
    data = pd.read_csv("abby/tests/datasets/click_impression_3_variants.csv")
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
        assert result.loc["click"]["mean_A"] == pytest.approx(0.45, rel=4)
        assert result.loc["click"]["mean_B"] == pytest.approx(0.63, rel=4)
        assert result.loc["click"]["var_A"] == pytest.approx(0.35101, rel=4)
        assert result.loc["click"]["var_B"] == pytest.approx(0.74050, rel=4)
        assert result.loc["click"]["absolute_difference"] == pytest.approx(0.18, rel=4)
        assert result.loc["click"]["lower_bound"] == pytest.approx(-0.02618, rel=4)
        assert result.loc["click"]["upper_bound"] == pytest.approx(0.38618, rel=4)
        assert result.loc["click"]["p_values"] == pytest.approx(0.08666, rel=4)

        name = "click/impression"
        assert result.loc[name]["mean_A"] == pytest.approx(0.135135, rel=4)
        assert result.loc[name]["mean_B"] == pytest.approx(0.162371, rel=4)
        assert result.loc[name]["var_A"] == pytest.approx(0.0295697, rel=4)
        assert result.loc[name]["var_B"] == pytest.approx(0.0475355, rel=4)
        assert result.loc[name]["absolute_difference"] == pytest.approx(0.027236, rel=4)
        assert result.loc[name]["lower_bound"] == pytest.approx(-0.027189, rel=4)
        assert result.loc[name]["upper_bound"] == pytest.approx(0.081661, rel=4)
        assert result.loc[name]["p_values"] == pytest.approx(0.326668, rel=4)

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


class TestCompareMultipleMultipleVariants:
    def test_multiple_result_multiple_variants(self, click_multi_df: Callable):
        result = compare_multiple(
            click_multi_df,
            ["control", "experiment_A", "experiment_B"],
            ["click", Ratio("click", "impression")],
        )
        control_and_A = result.loc[
            lambda df: (df["A"] == "control") & (df["B"] == "experiment_A")
        ]
        control_and_B = result.loc[
            lambda df: (df["A"] == "control") & (df["B"] == "experiment_B")
        ]
        assert control_and_A.loc["click"]["mean_A"] == pytest.approx(0.45, rel=4)
        assert control_and_A.loc["click"]["mean_B"] == pytest.approx(0.63, rel=4)
        assert control_and_A.loc["click"]["var_A"] == pytest.approx(0.35101, rel=4)
        assert control_and_A.loc["click"]["var_B"] == pytest.approx(0.74050, rel=4)
        assert control_and_A.loc["click"]["absolute_difference"] == pytest.approx(
            0.18, rel=4
        )
        assert control_and_A.loc["click"]["lower_bound"] == pytest.approx(
            -0.02618, rel=4
        )
        assert control_and_A.loc["click"]["upper_bound"] == pytest.approx(
            0.38618, rel=4
        )
        assert control_and_A.loc["click"]["p_values"] == pytest.approx(0.08666, rel=4)

        name = "click/impression"
        assert control_and_A.loc[name]["mean_A"] == pytest.approx(0.135135, rel=4)
        assert control_and_A.loc[name]["mean_B"] == pytest.approx(0.162371, rel=4)
        assert control_and_A.loc[name]["var_A"] == pytest.approx(0.0295697, rel=4)
        assert control_and_A.loc[name]["var_B"] == pytest.approx(0.0475355, rel=4)
        assert control_and_A.loc[name]["absolute_difference"] == pytest.approx(
            0.027236, rel=4
        )
        assert control_and_A.loc[name]["lower_bound"] == pytest.approx(-0.027189, rel=4)
        assert control_and_A.loc[name]["upper_bound"] == pytest.approx(0.081661, rel=4)
        assert control_and_A.loc[name]["p_values"] == pytest.approx(0.326668, rel=4)

        assert control_and_B.loc["click"]["mean_A"] == pytest.approx(0.45, rel=4)
        assert control_and_B.loc["click"]["mean_B"] == pytest.approx(0.98, rel=4)
        assert control_and_B.loc["click"]["var_A"] == pytest.approx(0.351010, rel=4)
        assert control_and_B.loc["click"]["var_B"] == pytest.approx(0.666263, rel=4)
        assert control_and_B.loc["click"]["absolute_difference"] == pytest.approx(
            0.53, rel=4
        )
        assert control_and_B.loc["click"]["lower_bound"] == pytest.approx(
            0.332315, rel=4
        )
        assert control_and_B.loc["click"]["upper_bound"] == pytest.approx(
            0.727685, rel=4
        )
        assert control_and_B.loc["click"]["p_values"] == pytest.approx(
            4.145101e-07, rel=4
        )
        name = "click/impression"
        assert control_and_B.loc[name]["mean_A"] == pytest.approx(0.135135, rel=4)
        assert control_and_B.loc[name]["mean_B"] == pytest.approx(0.259259, rel=4)
        assert control_and_B.loc[name]["var_A"] == pytest.approx(0.0295697, rel=4)
        assert control_and_B.loc[name]["var_B"] == pytest.approx(0.054965, rel=4)
        assert control_and_B.loc[name]["absolute_difference"] == pytest.approx(
            0.124124, rel=4
        )
        assert control_and_B.loc[name]["lower_bound"] == pytest.approx(0.067138, rel=4)
        assert control_and_B.loc[name]["upper_bound"] == pytest.approx(0.181111, rel=4)
        assert control_and_B.loc[name]["p_values"] == pytest.approx(0.00002, rel=4)
