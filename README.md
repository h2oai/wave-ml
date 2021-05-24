## Automatic Machine Learning (AutoML) for Wave Apps

This repository hosts Wave ML (`h2o-wave-ml`), a companion package for H2O Wave that makes it quick and easy to integrate AI/ML models into your applications.

Wave ML provides a simple, high-level API for training, deploying, scoring and explaining machine learning models, letting you build predictive and decision-support applications entirely in Python.

Wave ML runs on Linux, OSX, and Windows, and utilizes [H2O.ai's](https://h2o.ai) open-source [H2O](https://github.com/h2oai/h2o-3) and [AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) under the hood.

When Wave apps are run in [H2O AI Hybrid Cloud](https://www.h2o.ai/hybrid-cloud/) with GPU support, Wave ML optionally switches over to [Driverless AI](https://www.h2o.ai/products/h2o-driverless-ai/) for automatic feature engineering, machine learning, model deployment, and monitoring.

### Installation

Both Wave and Wave ML can be installed in tandem using pip:

```python3
> pip install h2o-wave[ml]
```

You can install Wave ML separately as well:

```python3
> pip install h2o-wave-ml
```

Note that the package pulled from *PyPI* doesn't have all the dependencies needed for **Cloud** development. *Steam* and *MLOps* packages are missing and have to be installed separately. However, they are present in the package from this repo in [release](https://github.com/h2oai/wave-ml/releases) section. Look into [setup.py](https://github.com/h2oai/wave-ml/blob/main/setup.py) to see the packages and include them in *requirements* file if needed.

### Quickstart

See a quickstart guide [here](https://github.com/h2oai/wave-ml/wiki/Quickstart).

<kbd><img src="assets/cm.gif" alt="confusion matrix"></kbd>

### API

The API can be found on official Wave page [here](https://wave.h2o.ai/docs/api/h2o_wave_ml/index).

### Development Setup

Check a Wiki for a [guide](https://github.com/h2oai/wave-ml/wiki/Developer).

### License

H2O Wave ML is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for more information.
