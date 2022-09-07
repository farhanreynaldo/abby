"""Compare module."""
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from abby.diagnose import sample_ratio_mismatch
from abby.utils import Ratio


def compare_multiple(
    data: pd.DataFrame, variants: List[str], metrics: List[Union[str, Ratio]]
):
    assert "variant_name" in data.columns, "Rename the variant column to `variant_name`"
    if sample_ratio_mismatch(data["variant_name"]) < 0.05:
        warnings.warn("There are sample ratio mismatch, your variants are not balance")

    ctrl, exp = variants

    results = dict()

    for metric in metrics:
        if isinstance(metric, Ratio):
            control_num = data.loc[
                data["variant_name"] == ctrl, metric.numerator
            ].values
            control_denom = data.loc[
                data["variant_name"] == ctrl, metric.denominator
            ].values
            exp_num = data.loc[data["variant_name"] == exp, metric.numerator].values
            exp_denom = data.loc[data["variant_name"] == exp, metric.denominator].values

            res_ratio = _compare_delta(control_num, control_denom, exp_num, exp_denom)
            results[metric.name] = res_ratio

        elif isinstance(metric, str):
            control_num = data.loc[data["variant_name"] == ctrl, metric]
            exp_num = data.loc[data["variant_name"] == exp, metric]

            results[metric] = _compare_ttest(
                control_num,
                exp_num,
            )
        else:
            raise ValueError(f"Unknown type: {type(metric)}")

    return results


def compare_ttest(
    data: pd.DataFrame,
    variants: List[str],
    numerator: str,
):
    assert "variant_name" in data.columns, "Rename the variant column to `variant_name`"
    if sample_ratio_mismatch(data["variant_name"]) < 0.05:
        warnings.warn("There are sample ratio mismatch, your variants are not balance")

    ctrl, exp = variants

    control_num = data.loc[data["variant_name"] == ctrl, numerator]
    exp_num = data.loc[data["variant_name"] == exp, numerator]

    return _compare_ttest(
        control_num,
        exp_num,
    )


def _compare_ttest(control: np.array, experiment: np.array) -> Dict[str, float]:
    control_size, exp_size = len(control), len(experiment)
    control_mean, exp_mean = np.mean(control), np.mean(experiment)

    control_var, exp_var = np.var(control, ddof=1), np.var(experiment, ddof=1)

    delta = exp_mean - control_mean
    _, p_values = stats.ttest_ind(control, experiment, equal_var=False)
    stde = 1.96 * np.sqrt(control_var / control_size + exp_var / exp_size)

    return dict(
        control_mean=control_mean,
        experiment_mean=exp_mean,
        control_var=control_var,
        experiment_var=exp_var,
        absolute_difference=delta,
        lower_bound=delta - stde,
        upper_bound=delta + stde,
        p_values=p_values,
    )


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
    denominator: Optional[str] = "",
) -> Dict[str, float]:
    assert "variant_name" in data.columns, "Rename the variant column to `variant_name`"
    if sample_ratio_mismatch(data["variant_name"]) < 0.05:
        warnings.warn("There are sample ratio mismatch, your variants are not balance")

    ctrl, exp = variants

    control_num = data.loc[data["variant_name"] == ctrl, numerator]
    control_denom = data.loc[data["variant_name"] == ctrl, denominator]
    exp_num = data.loc[data["variant_name"] == exp, numerator]
    exp_denom = data.loc[data["variant_name"] == exp, denominator]

    return _compare_delta(
        control_num,
        control_denom,
        exp_num,
        exp_denom,
    )


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
    control_num: np.array,
    control_denom: np.array,
    exp_num: np.array,
    exp_denom: np.array,
) -> Dict[str, float]:

    control_size = len(control_num)
    exp_size = len(exp_num)

    control_var = ratio_variance(control_num, control_denom)
    experiment_var = ratio_variance(exp_num, exp_denom)

    control_mean = control_num.sum() / control_denom.sum()
    experiment_mean = exp_num.sum() / exp_denom.sum()

    delta = experiment_mean - control_mean
    stde = 1.96 * np.sqrt(control_var / control_size + experiment_var / exp_size)

    z_scores = np.abs(delta) / np.sqrt(
        control_var / control_size + experiment_var / exp_size
    )
    p_values = stats.norm.sf(abs(z_scores)) * 2

    return dict(
        control_mean=control_mean,
        experiment_mean=experiment_mean,
        control_var=control_var,
        experiment_var=experiment_var,
        absolute_difference=experiment_mean - control_mean,
        lower_bound=delta - stde,
        upper_bound=delta + stde,
        p_values=p_values,
    )


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
