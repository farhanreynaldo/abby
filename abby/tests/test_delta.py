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
def click_multi_df() -> pd.DataFrame:
    data = pd.read_csv("abby/tests/datasets/click_impression_3_variants.csv")
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
        ).loc[0]
        assert result["mean_A"] == pytest.approx(0.135135, rel=4)
        assert result["mean_B"] == pytest.approx(0.162371, rel=4)
        assert result["var_A"] == pytest.approx(0.0295697, rel=4)
        assert result["var_B"] == pytest.approx(0.0475355, rel=4)
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


class TestCompareDeltaMultipleVariants:
    def test_delta_result_multiple_variants(self, click_multi_df: Callable):
        result = compare_delta(
            click_multi_df,
            ["control", "experiment_A", "experiment_B"],
            "click",
            "impression",
        )
        control_and_A = result.loc[
            lambda df: (df["A"] == "control") & (df["B"] == "experiment_A")
        ].loc[0]
        control_and_B = result.loc[
            lambda df: (df["A"] == "control") & (df["B"] == "experiment_B")
        ].loc[0]
        assert control_and_A["mean_A"] == pytest.approx(0.135135, rel=4)
        assert control_and_A["mean_B"] == pytest.approx(0.162371, rel=4)
        assert control_and_A["var_A"] == pytest.approx(0.0295697, rel=4)
        assert control_and_A["var_B"] == pytest.approx(0.0475355, rel=4)
        assert control_and_A["absolute_difference"] == pytest.approx(0.027236, rel=4)
        assert control_and_A["lower_bound"] == pytest.approx(-0.027189, rel=4)
        assert control_and_A["upper_bound"] == pytest.approx(0.081661, rel=4)
        assert control_and_A["p_values"] == pytest.approx(0.326668, rel=4)

        assert control_and_B["mean_A"] == pytest.approx(0.135135, rel=4)
        assert control_and_B["mean_B"] == pytest.approx(0.259259, rel=4)
        assert control_and_B["var_A"] == pytest.approx(0.0295697, rel=4)
        assert control_and_B["var_B"] == pytest.approx(0.054965, rel=4)
        assert control_and_B["absolute_difference"] == pytest.approx(0.124124, rel=4)
        assert control_and_B["lower_bound"] == pytest.approx(0.067138, rel=4)
        assert control_and_B["upper_bound"] == pytest.approx(0.181111, rel=4)
        assert control_and_B["p_values"] == pytest.approx(0.00002, rel=4)
