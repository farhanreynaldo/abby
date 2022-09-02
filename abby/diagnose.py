import pandas as pd
from scipy import stats


def sample_ratio_mismatch(series: pd.Series) -> float:
    observed = series.value_counts().values
    expected = [len(series) / 2, len(series) / 2]
    pvalue = stats.chisquare(observed, expected).pvalue
    return pvalue


def check_null():
    raise NotImplementedError
