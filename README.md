# Automatic Machine Learning (AutoML) for Wave Apps

This repository hosts Wave ML (`h2o-wave-ml`), a companion package for H2O Wave that makes it quick and easy to integrate AI/ML models into your applications.

Wave ML provides a simple, high-level API for training, deploying, scoring and explaining machine learning models, letting you build predictive and decision-support applications entirely in Python.

Wave ML runs on Linux, OSX, and Windows, and utilizes [H2O.ai's](https://h2o.ai) open-source [H2O](https://github.com/h2oai/h2o-3) and [AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) under the hood.

When Wave apps are run in [H2O AI Hybrid Cloud](https://www.h2o.ai/hybrid-cloud/) with GPU support, Wave ML optionally switches over to [Driverless AI](https://www.h2o.ai/products/h2o-driverless-ai/) for automatic feature engineering, machine learning, model deployment, and monitoring.


## Quickstart

**The API is under development and is not stable.**

(TODO: `pip install` instructions)

(TODO: Show source code for a small, but complete, app that trains on, say, creditcard.csv, and displays a UI that let's you tweak a couple of fields and get predictions.)

# API

### build_model()

```python3
def build_model(filename, target, metric = WaveModelMetric.AUTO, model_backend_type = None, **kwargs):
```

Builds a model. If `model_backend_type` not specified the function will determine correct backend based on a current environment.

- `filename`A string containing the filename to a dataset.
- `target`: A name of the target column.
- `metric`: Optional metric to be used during the building process specified by `h2o_wave_ml.WaveModelMetric`.
- `model_backend_type`: Optional backend model type specified by `h2o_wave_ml.WaveModelBackendType`.
- `kwargs`: Optional parameters passed to the backend.

Returns:
  A wave model.

### get_model()

```python3
def get_model(id_, model_type = None):
```

Get a model accessible on a backend.


- `id_`: Identification of a model.
- `model_type`: Optional model backend type specified by `h2o_wave_ml_WaveModelType`.

Returns:
    A wave model.

### save_model()

```python3
def save_model(backend, folder):
```

Save a model to disk. Works only for an `H2O 3` backend.

- `backend`: A model backend produced by build_model.
- `folder`: A directory where the saved model will be put to.

Returns:
    Path to a saved model.

### load_model()

```python3
def load_model(filename):
```

Load a model from disk into the instance. This will always return a model with an `H2O 3` backend.

- `filename`: Path to saved model.

Returns:
    A wave model.
    
# Environment variables

The environment variables ensure the correct behaviour of a function calls behind the scenes. Based on a setup the API might spawn a new `H2O 3` instance or use existing `DAI` instance, use plain password to authenticate or utilize OpenID Connect, etc.

Currently just one environment variable is available to set:

- `H2O_WAVE_H2O3_URL`, if set the existing instance of `H2O 3` will be used instead of spawning a fresh one.

# Development setup 

A python of version `3.6.1` or greater is required.

1. Clone repo
2. Type `make setup`

# Examples

To build a model a dataset in `.csv` format is needed and target column needs to be specified:

```python
from h2o_wave.ml import build_model

train_set = './creditcard_train.csv'
model = build_model(train_set, target='DEFAULT_PAYMENT_NEXT_MONTH')
```

Once the model is built we can make a predictions on training dataset:

```python
from h2o_wave.ml import build_model

test_set = './creditcard_test.csv'
train_set = './creditcard_train.csv'

model = build_model(train_set, target='DEFAULT_PAYMENT_NEXT_MONTH')
predictions = model.predict(test_set)
```

or store model onto disk. The resulting model file path is returned by the `save_model()` function call:

```python
from h2o_wave.ml import build_model, save_model

train_set = './creditcard_train.csv'
model = build_model(train_set, target='DEFAULT_PAYMENT_NEXT_MONTH')
path = save_model(model)
```

If model stored, we can load it up and make predictions:

```python
from h2o_wave.ml import load_model

model = load_model('./MyModel')
predictions = model.predict('./Datasets/creditcard_test_cat.csv')
```

The `.predict()` method call can take either the file path or python object with a raw data. Column names need to be specified omitting the target column. The example shows prediction on one row:

```python
from h2o_wave.ml import load_model

data = [["ID", "LIMIT_BAL", "SEX"], [24001, 50000, "male"]]

model = load_model('./MyModel')
predictions = model.predict(data)
```