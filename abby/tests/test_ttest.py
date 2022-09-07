from typing import Callable

import numpy as np
import pandas as pd
import pytest

from abby.compare import compare_ttest


@pytest.fixture
def sleeps() -> pd.DataFrame:
    data = pd.read_csv("abby/tests/datasets/sleeps.csv").assign(
        variant_name=lambda df: df["group"].map({1: "a", 2: "b"})
    )
    return data


@pytest.fixture
def sleeps_wrong(sleeps):
    return sleeps.rename(columns={"variant_name": "group"})


@pytest.fixture
def sleeps_mismatch_ratio(sleeps: Callable) -> pd.DataFrame:
    return sleeps.assign(variant_name=np.repeat(["a", "b"], [18, 2]))


@pytest.fixture
def click_multi_df() -> pd.DataFrame:
    data = pd.read_csv("abby/tests/datasets/click_impression_3_variants.csv")
    return data


class TestCompareTtest:
    def test_ttest_result(self, sleeps: Callable):
        result = compare_ttest(sleeps, ["a", "b"], "extra").loc[0]
        assert result["mean_A"] == pytest.approx(0.75, rel=4)
        assert result["mean_B"] == pytest.approx(2.33, rel=4)
        assert result["var_A"] == pytest.approx(3.200556, rel=4)
        assert result["var_B"] == pytest.approx(4.009000, rel=4)
        assert result["absolute_difference"] == pytest.approx(1.5800, rel=4)
        assert result["lower_bound"] == pytest.approx(-0.2054832, rel=4)
        assert result["upper_bound"] == pytest.approx(3.3654832, rel=4)
        assert result["p_values"] == pytest.approx(0.07939, rel=4)

    def test_ttest_wrong_variant_column_name(self, sleeps_wrong: Callable):
        with pytest.raises(AssertionError):
            compare_ttest(sleeps_wrong, ["a", "b"], "extra")

    def test_ttest_raise_warning_mismatch_ratio(self, sleeps_mismatch_ratio: Callable):
        with pytest.warns(UserWarning):
            compare_ttest(sleeps_mismatch_ratio, ["a", "b"], "extra")


class TestCompareTtestMultipleVariants:
    def test_ttest_result_multiple_variants(self, click_multi_df: Callable):
        result = compare_ttest(
            click_multi_df, ["control", "experiment_A", "experiment_B"], "click"
        )
        control_and_A = result.loc[
            lambda df: (df["A"] == "control") & (df["B"] == "experiment_A")
        ].loc[0]
        control_and_B = result.loc[
            lambda df: (df["A"] == "control") & (df["B"] == "experiment_B")
        ].loc[0]
        assert control_and_A["mean_A"] == pytest.approx(0.45, rel=4)
        assert control_and_A["mean_B"] == pytest.approx(0.63, rel=4)
        assert control_and_A["var_A"] == pytest.approx(0.351010, rel=4)
        assert control_and_A["var_B"] == pytest.approx(0.740505, rel=4)
        assert control_and_A["absolute_difference"] == pytest.approx(0.18, rel=4)
        assert control_and_A["lower_bound"] == pytest.approx(-0.024772, rel=4)
        assert control_and_A["upper_bound"] == pytest.approx(0.384772, rel=4)
        assert control_and_A["p_values"] == pytest.approx(8.666844e-02, rel=4)

        assert control_and_B["mean_A"] == pytest.approx(0.45, rel=4)
        assert control_and_B["mean_B"] == pytest.approx(0.98, rel=4)
        assert control_and_B["var_A"] == pytest.approx(0.351010, rel=4)
        assert control_and_B["var_B"] == pytest.approx(0.666263, rel=4)
        assert control_and_B["absolute_difference"] == pytest.approx(0.53, rel=4)
        assert control_and_B["lower_bound"] == pytest.approx(0.332315, rel=4)
        assert control_and_B["upper_bound"] == pytest.approx(0.727685, rel=4)
        assert control_and_B["p_values"] == pytest.approx(4.145101e-07, rel=4)
