# abby

> A/B testing for Human

Abby is a A/B testing library package for human. Abby aims to make A/B testing as easy as ABC and accessible to anyone.

[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]

# Installation

```sh
$ pip install abby
```

# Quick Start

Please note that your variant name column should be named as `variant_name`. Otherwise, the method will raise an error and ask you to adjust the column name accordingly.

## A/B testing for continuous metric

```python
from abby.datasets import load_dataset
from abby import compare_ttest

data = load_dataset('click_impression')

compare_ttest(data=data, variants=['control', 'experiment'], numerator='click')
```

## A/B testing for ratio metric

```python
from abby.datasets import load_dataset
from abby import compare_delta

data = load_dataset("click_impression")

compare_delta(
    data=data,
    variants=["control", "experiment"],
    numerator="click",
    denominator="impression",
)
```

## A/B testing for ratio metric using bootstrap

```python
from abby.datasets import load_dataset
from abby import compare_bootstrap_delta

data = load_dataset("click_impression")

compare_bootstrap_delta(
    data=data,
    variants=["control", "experiment"],
    numerator="click",
    denominator="impression",
    n_bootstrap=10_000,
)
```

## A/B testing for multiple metrics

```python
from abby.datasets import load_dataset
from abby import compare_multiple, Ratio

data = load_dataset("click_impression")

result = compare_multiple(
    data=data,
    variants=["control", "experiment"],
    metrics=["click", Ratio("click", "impression")],
)
```

<!-- Badges -->

[pypi-image]: https://img.shields.io/pypi/v/abby
[pypi-url]: https://pypi.org/project/abby/
[build-image]: https://github.com/farhanreynaldo/abby/actions/workflows/test.yml/badge.svg
[build-url]: https://github.com/farhanreynaldo/abby/actions/workflows/test.yml
