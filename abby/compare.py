"""Compare module."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


def compare_ttest(
    data: pd.DataFrame,
    variants: List[str],
    numerator: str,
    **kwargs,
):
    assert "variant_name" in data.columns, "Rename the variant column to `variant_name`"

    ctrl, exp = variants

    numerator_ctrl = data.loc[data["variant_name"] == ctrl, numerator]
    numerator_exp = data.loc[data["variant_name"] == exp, numerator]

    return _compare_ttest(
        numerator_ctrl,
        numerator_exp,
    )


def _compare_ttest(control, experiment):

    control_size = len(control)
    exp_size = len(experiment)
    control_mean = np.mean(control)
    exp_mean = np.mean(experiment)

    control_var = np.var(control, ddof=1)
    exp_var = np.var(experiment, ddof=1)

    delta = exp_mean - control_mean

    df, se = _unequal_var_ttest_denom(control_var, control_size, exp_var, exp_size)

    _, p_values = stats.ttest_ind(control, experiment, equal_var=False)

    # upper and lower bounds
    lower_bound = delta - stats.t.ppf(0.975, df) * se
    upper_bound = delta + stats.t.ppf(0.975, df) * se

    return dict(
        control_mean=control_mean,
        experiment_mean=exp_mean,
        control_var=control_var,
        experiment_var=exp_var,
        absolute_difference=delta,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
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

    ctrl, exp = variants

    numerator_ctrl = data.loc[data["variant_name"] == ctrl, numerator].values
    denominator_ctrl = data.loc[data["variant_name"] == ctrl, denominator].values
    numerator_exp = data.loc[data["variant_name"] == exp, numerator].values
    denominator_exp = data.loc[data["variant_name"] == exp, denominator].values

    return _compare_bootstrap_delta(
        numerator_ctrl,
        denominator_ctrl,
        numerator_exp,
        denominator_exp,
        **kwargs,
    )


def compare_delta(
    data: pd.DataFrame,
    variants: List[str],
    numerator: str,
    denominator: Optional[str] = "",
) -> Dict[str, float]:
    assert "variant_name" in data.columns, "Rename the variant column to `variant_name`"

    ctrl, exp = variants

    numerator_ctrl = data.loc[data["variant_name"] == ctrl, numerator]
    denominator_ctrl = data.loc[data["variant_name"] == ctrl, denominator]
    numerator_exp = data.loc[data["variant_name"] == exp, numerator]
    denominator_exp = data.loc[data["variant_name"] == exp, denominator]

    return _compare_delta(
        numerator_ctrl, denominator_ctrl, numerator_exp, denominator_exp
    )


def _compare_bootstrap_delta(
    numerator_ctrl: np.array,
    denominator_ctrl: np.array,
    numerator_exp: np.array,
    denominator_exp: np.array,
    n_bootstrap: int = 10_000,
):
    n_users_a, n_users_b = len(numerator_ctrl), len(numerator_exp)
    n_users = n_users_a + n_users_b
    bs_observed = []

    for _ in tqdm(range(n_bootstrap)):
        conversion = np.hstack((numerator_ctrl, numerator_exp))
        session = np.hstack((denominator_ctrl, denominator_exp))

        assignments = np.random.choice(n_users, n_users, replace=True)
        ctrl_idxs = assignments[: int(n_users / 2)]
        test_idxs = assignments[int(n_users / 2) :]

        bs_denominator_ctrl = session[ctrl_idxs]
        bs_denominator_exp = session[test_idxs]
        bs_numerator_ctrl = conversion[ctrl_idxs]
        bs_numerator_exp = conversion[test_idxs]

        bs_observed.append(
            bs_numerator_exp.sum() / bs_denominator_exp.sum()
            - bs_numerator_ctrl.sum() / bs_denominator_ctrl.sum()
        )

    observed_diffs = (
        numerator_exp.sum() / denominator_exp.sum()
        - numerator_ctrl.sum() / denominator_ctrl.sum()
    )

    lower_bound, upper_bound = _confidence_interval_bootstrap(
        numerator_ctrl,
        denominator_ctrl,
        numerator_exp,
        denominator_exp,
        n_bootstrap,
    )
    p_values = 2 * (1 - (np.abs(observed_diffs) > np.array(bs_observed)).mean())
    return dict(
        control_mean=numerator_ctrl.sum() / denominator_ctrl.sum(),
        experiment_mean=numerator_exp.sum() / denominator_exp.sum(),
        control_var=ratio_variance(numerator_ctrl, denominator_ctrl),
        experiment_var=ratio_variance(numerator_exp, denominator_exp),
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
    numerator_ctrl: np.array,
    denominator_ctrl: np.array,
    numerator_exp: np.array,
    denominator_exp: np.array,
) -> Dict[str, float]:
    var_ctrl = ratio_variance(numerator_ctrl, denominator_ctrl)
    var_exp = ratio_variance(numerator_exp, denominator_exp)

    control_mean = numerator_ctrl.sum() / denominator_ctrl.sum()
    experiment_mean = numerator_exp.sum() / denominator_exp.sum()

    diff = experiment_mean - control_mean
    stde = 1.96 * np.sqrt(var_ctrl + var_exp)

    z_scores = np.abs(diff) / np.sqrt(var_ctrl + var_exp)
    p_values = stats.norm.sf(abs(z_scores)) * 2

    return dict(
        control_mean=control_mean,
        experiment_mean=experiment_mean,
        control_var=var_ctrl,
        experiment_var=var_exp,
        absolute_difference=experiment_mean - control_mean,
        lower_bound=diff - stde,
        upper_bound=diff + stde,
        p_values=p_values,
    )


def ratio_variance(num: np.array, denom: np.array) -> float:
    assert len(num) == len(
        denom
    ), f"Different length between num: {len(num)} and denom: {len(denom)}"
    n_users = len(num)
    denom_mean = np.mean(denom)
    num_mean = np.mean(num)
    denom_variance = np.var(denom, ddof=1)
    num_variance = np.var(num, ddof=1)
    denom_num_covariance = np.cov(denom, num, ddof=1)[0][1]
    return (
        (num_variance) / (denom_mean**2)
        - 2 * num_mean * denom_num_covariance / (denom_mean**3)
        + (num_mean**2) * (denom_variance) / (denom_mean**4)
    ) / n_users


def _unequal_var_ttest_denom(v1, n1, v2, n2):
    vn1 = v1 / n1
    vn2 = v2 / n2
    with np.errstate(divide="ignore", invalid="ignore"):
        df = (vn1 + vn2) ** 2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))

    # If df is undefined, variances are zero (assumes n1 > 0 & n2 > 0).
    # Hence it doesn't matter what df is as long as it's not NaN.
    df = np.where(np.isnan(df), 1, df)
    denom = np.sqrt(vn1 + vn2)
    return df, denom
