"""Compare module."""
from typing import List, Optional

import numpy as np
import scipy
from tqdm import tqdm
import pandas as pd


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
):
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
    conversions_ctrl: np.array,
    sessions_ctrl: np.array,
    conversions_exp: np.array,
    sessions_exp: np.array,
):
    n_users_a, n_users_b = len(conversions_ctrl), len(conversions_exp)

    mean_sessions_ctrl = sessions_ctrl.mean()
    mean_conversions_ctrl = conversions_ctrl.mean()
    var_sessions_ctrl = sessions_ctrl.var()
    var_conversions_ctrl = conversions_ctrl.var()

    mean_sessions_exp = sessions_exp.mean()
    mean_conversions_exp = conversions_exp.mean()
    var_sessions_exp = sessions_exp.var()
    var_conversions_exp = conversions_exp.var()

    cov_ctrl = (
        (conversions_ctrl - mean_conversions_ctrl)
        * (sessions_ctrl - mean_sessions_ctrl)
    ).mean()
    cov_exp = (
        (conversions_exp - mean_conversions_exp) * (sessions_exp - mean_sessions_exp)
    ).mean()

    var_ctrl = (
        var_conversions_ctrl / mean_sessions_ctrl**2
        + var_sessions_ctrl * mean_conversions_ctrl**2 / mean_sessions_ctrl**4
        - 2 * mean_conversions_ctrl / mean_sessions_ctrl**3 * cov_ctrl
    )
    var_exp = (
        var_conversions_exp / mean_sessions_exp**2
        + var_sessions_exp * mean_conversions_exp**2 / mean_sessions_exp**4
        - 2 * mean_conversions_exp / mean_sessions_exp**3 * cov_exp
    )

    cvrs_ctrl = conversions_ctrl.sum() / sessions_ctrl.sum()
    cvrs_exp = conversions_exp.sum() / sessions_exp.sum()

    z_scores = np.abs(cvrs_exp - cvrs_ctrl) / np.sqrt(
        var_ctrl / n_users_a + var_exp / n_users_b
    )
    p_values = 2 * (1 - scipy.stats.norm(loc=0, scale=1).cdf(z_scores))
    return p_values
