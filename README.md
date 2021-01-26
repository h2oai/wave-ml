# Essential ML API for Wave

## Development installation

1. Clone repo
2. Type `make setup`

## Examples

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