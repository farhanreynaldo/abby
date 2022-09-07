"""Compare module."""
import warnings
from itertools import combinations
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from abby.diagnose import sample_ratio_mismatch
from abby.utils import Ratio


def compare_multiple(
    data: pd.DataFrame,
    variants: List[str],
    metrics: List[Union[str, Ratio]],
    method: Optional[str] = "bonferroni",
):
    assert "variant_name" in data.columns, "Rename the variant column to `variant_name`"
    assert set(variants) == set(
        data["variant_name"].unique()
    ), "One or more variants are not found in the data"
    if sample_ratio_mismatch(data["variant_name"]) < 0.05:
        warnings.warn("There are sample ratio mismatch, your variants are not balance")

    results = []

    for metric in metrics:
        if isinstance(metric, Ratio):
            print("ratio")
            result_df = compare_delta(
                data, variants, metric.numerator, metric.denominator, method=method
            )
            result_df["metric"] = metric.name
            results.append(result_df)

        elif isinstance(metric, str):
            result_df = compare_ttest(data, variants, metric, method=method)
            result_df["metric"] = metric
            results.append(result_df)
        else:
            raise ValueError(f"Unknown type: {type(metric)}")

    return pd.concat(results).set_index("metric")


def compare_ttest(
    data: pd.DataFrame,
    variants: List[str],
    numerator: str,
    method: Optional[str] = "bonferroni",
):
    assert "variant_name" in data.columns, "Rename the variant column to `variant_name`"
    assert set(variants) == set(
        data["variant_name"].unique()
    ), "One or more variants are not found in the data"
    if sample_ratio_mismatch(data["variant_name"]) < 0.05:
        warnings.warn("There are sample ratio mismatch, your variants are not balance")

    if len(variants) == 2:
        a_name, b_name = variants

        a_num = data.loc[data["variant_name"] == a_name, numerator]
        b_num = data.loc[data["variant_name"] == b_name, numerator]

        return _compare_ttest(a_num, a_name, b_num, b_name)

    elif len(variants) > 2:
        pairs = combinations(variants, 2)
        results = []
        for (a_name, b_name) in pairs:
            a_num = data.loc[data["variant_name"] == a_name, numerator]
            b_num = data.loc[data["variant_name"] == b_name, numerator]
            results.append(_compare_ttest(a_num, a_name, b_num, b_name))
        results_df = pd.concat(results)
        results_df["p_values_corrected"] = multipletests(
            results_df["p_values"].values, method=method
        )[1]
        return results_df


def _compare_ttest(
    a: np.array, a_name: str, b: np.array, b_name: str
) -> Dict[str, float]:
    a_size, b_size = len(a), len(b)
    a_mean, b_mean = np.mean(a), np.mean(b)

    a_var, b_var = np.var(a, ddof=1), np.var(b, ddof=1)

    delta = b_mean - a_mean
    _, p_values = stats.ttest_ind(a, b, equal_var=False)
    stde = 1.96 * np.sqrt(a_var / a_size + b_var / b_size)

    result = dict(
        A=a_name,
        B=b_name,
        mean_A=a_mean,
        mean_B=b_mean,
        var_A=a_var,
        var_B=b_var,
        absolute_difference=delta,
        lower_bound=delta - stde,
        upper_bound=delta + stde,
        p_values=p_values,
    )
    return pd.DataFrame([result])


def compare_bootstrap_delta(
    data: pd.DataFrame,
    variants: List[str],
    numerator: str,
    denominator: Optional[str] = "",
    **kwargs,
):
    assert "variant_name" in data.columns, "Rename the variant column to `variant_name`"
    if sample_ratio_mismatch(data["variant_name"]) < 0.05:
        warnings.warn("There are sample ratio mismatch, your variants are not balance")

    ctrl, exp = variants

    control_num = data.loc[data["variant_name"] == ctrl, numerator].values
    control_denom = data.loc[data["variant_name"] == ctrl, denominator].values
    exp_num = data.loc[data["variant_name"] == exp, numerator].values
    exp_denom = data.loc[data["variant_name"] == exp, denominator].values

    return _compare_bootstrap_delta(
        control_num,
        control_denom,
        exp_num,
        exp_denom,
        **kwargs,
    )


def compare_delta(
    data: pd.DataFrame,
    variants: List[str],
    numerator: str,
    denominator: str,
    method: Optional[str] = "bonferroni",
) -> Dict[str, float]:
    assert "variant_name" in data.columns, "Rename the variant column to `variant_name`"
    assert set(variants) == set(
        data["variant_name"].unique()
    ), "One or more variants are not found in the data"
    if sample_ratio_mismatch(data["variant_name"]) < 0.05:
        warnings.warn("There are sample ratio mismatch, your variants are not balance")

    if len(variants) == 2:
        a_name, b_name = variants

        a_num = data.loc[data["variant_name"] == a_name, numerator]
        a_denom = data.loc[data["variant_name"] == a_name, denominator]
        b_num = data.loc[data["variant_name"] == b_name, numerator]
        b_denom = data.loc[data["variant_name"] == b_name, denominator]

        return _compare_delta(a_num, a_denom, a_name, b_num, b_denom, b_name)

    elif len(variants) > 2:
        pairs = combinations(variants, 2)
        results = []
        for (a_name, b_name) in pairs:
            a_num = data.loc[data["variant_name"] == a_name, numerator]
            a_denom = data.loc[data["variant_name"] == a_name, denominator]
            b_num = data.loc[data["variant_name"] == b_name, numerator]
            b_denom = data.loc[data["variant_name"] == b_name, denominator]
            results.append(
                _compare_delta(
                    a_num,
                    a_denom,
                    a_name,
                    b_num,
                    b_denom,
                    b_name,
                )
            )
        results_df = pd.concat(results)
        results_df["p_values_corrected"] = multipletests(
            results_df["p_values"].values, method=method
        )[1]
        return results_df


def _compare_bootstrap_delta(
    control_num: np.array,
    control_denom: np.array,
    exp_num: np.array,
    exp_denom: np.array,
    n_bootstrap: int = 10_000,
):
    n_users_a, n_users_b = len(control_num), len(exp_num)
    n_users = n_users_a + n_users_b
    bs_observed = []

    for _ in tqdm(range(n_bootstrap)):
        conversion = np.hstack((control_num, exp_num))
        session = np.hstack((control_denom, exp_denom))

        assignments = np.random.choice(n_users, n_users, replace=True)
        ctrl_idxs = assignments[: int(n_users / 2)]
        test_idxs = assignments[int(n_users / 2) :]

        bs_control_denom = session[ctrl_idxs]
        bs_denominator_exp = session[test_idxs]
        bs_control_num = conversion[ctrl_idxs]
        bs_exp_num = conversion[test_idxs]

        bs_observed.append(
            bs_exp_num.sum() / bs_denominator_exp.sum()
            - bs_control_num.sum() / bs_control_denom.sum()
        )

    observed_diffs = (
        exp_num.sum() / exp_denom.sum() - control_num.sum() / control_denom.sum()
    )

    lower_bound, upper_bound = _confidence_interval_bootstrap(
        control_num,
        control_denom,
        exp_num,
        exp_denom,
        n_bootstrap,
    )
    p_values = 2 * (1 - (np.abs(observed_diffs) > np.array(bs_observed)).mean())
    return dict(
        control_mean=control_num.sum() / control_denom.sum(),
        experiment_mean=exp_num.sum() / exp_denom.sum(),
        control_var=ratio_variance(control_num, control_denom),
        experiment_var=ratio_variance(exp_num, exp_denom),
        absolute_difference=observed_diffs,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        p_values=p_values,
    )


def _confidence_interval_bootstrap(
    numerator_ctrl: np.array,
    denominator_ctrl: np.array,
    numerator_exp: np.array,
    denominator_exp: np.array,
    n_bootstrap: int,
):
    bs_observed = []

    for _ in tqdm(range(n_bootstrap)):
        ctrl_idxs = np.random.choice(
            len(numerator_ctrl), len(numerator_ctrl), replace=True
        )
        exp_idxs = np.random.choice(
            len(numerator_exp), len(numerator_exp), replace=True
        )

        bs_denominator_ctrl = denominator_ctrl[ctrl_idxs]
        bs_denominator_exp = denominator_exp[exp_idxs]
        bs_numerator_ctrl = numerator_ctrl[ctrl_idxs]
        bs_numerator_exp = numerator_exp[exp_idxs]

        bs_observed.append(
            bs_numerator_exp.sum() / bs_denominator_exp.sum()
            - bs_numerator_ctrl.sum() / bs_denominator_ctrl.sum()
        )
    return np.percentile(bs_observed, [2.5, 97.5])


def _compare_delta(
    a_num: np.array,
    a_denom: np.array,
    a_name: str,
    b_num: np.array,
    b_denom: np.array,
    b_name: str,
) -> Dict[str, float]:

    a_size = len(a_num)
    b_size = len(b_num)

    a_var = ratio_variance(a_num, a_denom)
    b_var = ratio_variance(b_num, b_denom)

    a_mean = a_num.sum() / a_denom.sum()
    b_mean = b_num.sum() / b_denom.sum()

    delta = b_mean - a_mean
    stde = 1.96 * np.sqrt(a_var / a_size + b_var / b_size)

    z_scores = np.abs(delta) / np.sqrt(a_var / a_size + b_var / b_size)
    p_values = stats.norm.sf(abs(z_scores)) * 2

    result = dict(
        A=a_name,
        B=b_name,
        mean_A=a_mean,
        mean_B=b_mean,
        var_A=a_var,
        var_B=b_var,
        absolute_difference=delta,
        lower_bound=delta - stde,
        upper_bound=delta + stde,
        p_values=p_values,
    )
    return pd.DataFrame([result])


def ratio_variance(num: np.array, denom: np.array) -> float:
    """
    Reference:
    https://www.stat.cmu.edu/~hseltman/files/ratio.pdf
    """
    assert len(num) == len(
        denom
    ), f"Different length between num: {len(num)} and denom: {len(denom)}"
    denom_mean = np.mean(denom)
    num_mean = np.mean(num)
    denom_variance = np.var(denom, ddof=1)
    num_variance = np.var(num, ddof=1)
    denom_num_covariance = np.cov(denom, num, ddof=1)[0][1]
    return (
        (num_variance) / (denom_mean**2)
        - 2 * num_mean * denom_num_covariance / (denom_mean**3)
        + (num_mean**2) * (denom_variance) / (denom_mean**4)
    )
