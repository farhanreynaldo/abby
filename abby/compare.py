"""Compare module."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


def compare_bootstrap(
    data: pd.DataFrame,
    variants: List[str],
    numerator: str,
    denominator: Optional[str] = "",
    **kwargs,
):
    assert "variant_name" in data.columns, "Rename the variant column to `variant_name`"

    ctrl, exp = variants

    numerator_ctrl = data.loc[data["variant_name"] == ctrl, numerator]
    denominator_ctrl = data.loc[data["variant_name"] == ctrl, denominator]
    numerator_exp = data.loc[data["variant_name"] == exp, numerator]
    denominator_exp = data.loc[data["variant_name"] == exp, denominator]

    return _compare_bootstrap(
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


def _compare_bootstrap(
    conversions_ctrl: np.array,
    sessions_ctrl: np.array,
    conversions_exp: np.array,
    sessions_exp: np.array,
    n_bootstrap: int = 10_000,
):
    n_users_a, n_users_b = len(conversions_ctrl), len(conversions_exp)
    n_users = n_users_a + n_users_b
    bs_observed = []

    for _ in tqdm(range(n_bootstrap)):
        conversion = np.hstack((conversions_ctrl, conversions_exp))
        session = np.hstack((sessions_ctrl, sessions_exp))

        assignments = np.random.choice(n_users, n_users, replace=True)
        ctrl_idxs = assignments[: int(n_users / 2)]
        test_idxs = assignments[int(n_users / 2) :]

        bs_sessions_ctrl = session[ctrl_idxs]
        bs_sessions_exp = session[test_idxs]
        bs_conversions_ctrl = conversion[ctrl_idxs]
        bs_conversions_exp = conversion[test_idxs]

        bs_observed.append(
            bs_conversions_exp.sum() / bs_sessions_exp.sum()
            - bs_conversions_ctrl.sum() / bs_sessions_ctrl.sum()
        )

    observed_diffs = (
        conversions_exp.sum() / sessions_exp.sum()
        - conversions_ctrl.sum() / sessions_ctrl.sum()
    )
    p_values = 2 * (1 - (np.abs(observed_diffs) > np.array(bs_observed)).mean())
    return p_values


def _compare_delta(
    numerator_ctrl: np.array,
    denominator_ctrl: np.array,
    numerator_exp: np.array,
    denominator_exp: np.array,
) -> Dict[str, float]:
    var_ctrl = ratio_variance(numerator_ctrl, denominator_ctrl)
    var_exp = ratio_variance(numerator_exp, denominator_exp)

    cvrs_ctrl = numerator_ctrl.sum() / denominator_ctrl.sum()
    cvrs_exp = numerator_exp.sum() / denominator_exp.sum()

    diff = cvrs_exp - cvrs_ctrl
    stde = 1.96 * np.sqrt(var_ctrl + var_exp)

    z_scores = np.abs(diff) / np.sqrt(var_ctrl + var_exp)
    p_values = stats.norm.sf(abs(z_scores)) * 2

    return dict(
        control_mean=cvrs_ctrl,
        experiment_mean=cvrs_exp,
        control_var=var_ctrl,
        experiment_var=var_exp,
        absolute_difference=cvrs_exp - cvrs_ctrl,
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
