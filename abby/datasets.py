import pandas as pd


def load_dataset(name: str, **kwargs) -> pd.DataFrame:
    url = f"https://raw.githubusercontent.com/farhanreynaldo/abby/main/abby/tests/datasets/{name}.csv"  # noqa

    df = pd.read_csv(url, **kwargs)
    return df
