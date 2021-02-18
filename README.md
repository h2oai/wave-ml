## Automatic Machine Learning (AutoML) for Wave Apps

This repository hosts Wave ML (`h2o-wave-ml`), a companion package for H2O Wave that makes it quick and easy to integrate AI/ML models into your applications.

Wave ML provides a simple, high-level API for training, deploying, scoring and explaining machine learning models, letting you build predictive and decision-support applications entirely in Python.

Wave ML runs on Linux, OSX, and Windows, and utilizes [H2O.ai's](https://h2o.ai) open-source [H2O](https://github.com/h2oai/h2o-3) and [AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) under the hood.

When Wave apps are run in [H2O AI Hybrid Cloud](https://www.h2o.ai/hybrid-cloud/) with GPU support, Wave ML optionally switches over to [Driverless AI](https://www.h2o.ai/products/h2o-driverless-ai/) for automatic feature engineering, machine learning, model deployment, and monitoring.

## Quickstart

The package can be installed using `pip`:

```shell script
pip install h2o-wave-ml
```

or along with H2O Wave:

```shell script
pip install h2o-wave[ml]
```

<kbd><img src="assets/cm.gif" alt="confusion matrix"></kbd>

```python
"""
Take a Titanic dataset, train a model and show a confusion matrix based on that model.
"""

import datatable as dt
from h2o_wave import main, app, Q, ui
from h2o_wave_ml import build_model
from sklearn.metrics import confusion_matrix

dataset = './titanic.csv'
target_column = 'Survived'

# Train model and make a prediction
model = build_model(dataset, target_column=target_column)
prediction = model.predict(file_path=dataset)

# Prepare the `actual` values from target_column
df = dt.fread(dataset)
y_true = df[target_column].to_list()[0]

template = '''
## Confusion Matrix for Titanic
| | | |
|:-:|:-:|:-:|
| | Survived | Not survived |
| Survived | **{tp}** | {fp} (FP) |
| Not survived | {fn} (FN) | **{tn}** |
<br><br>
'''


@app('/demo')
async def serve(q: Q):

    # Get a threshold value if available or 0.5 by default
    threshold = q.args.slider if 'slider' in q.args else 0.5

    # Compute confusion matrix
    y_pred = [p[1] < threshold for p in prediction]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Handle interaction
    if not q.client.initialized:  # First visit, create a card for the matrix
        q.page['matrix'] = ui.form_card(box='1 1 3 4', items=[
            ui.text(template.format(tn=tn, fp=fp, fn=fn, tp=tp)),
            ui.slider(name='slider', label='Threshold', min=0, max=1, step=0.01, value=0.5,
                      trigger=True),
        ])
        q.client.initialized = True
    else:
        q.page['matrix'].items[0].text.content = template.format(tn=tn, fp=fp, fn=fn, tp=tp)
        q.page['matrix'].items[1].slider.value = threshold

    await q.page.save()
```

## API

**The API is under development and is not stable.**

### build_model()

```python3
def build_model(file_path: str, target_column: str, model_metric: ModelMetric = ModelMetric.AUTO,
                model_type: Optional[ModelType] = None, **kwargs) -> Model:
```

Trains a model.

If `model_type` is not specified, it is inferred from the current environment. Defaults to a `H2O-3` model.

- `file_path`: The path to the training dataset.
- `target_column`: The name of the target column (the column to be predicted).
- `model_metric`: Optional evaluation metric to be used during modeling, specified by `h2o_wave_ml.ModelMetric`.
- `model_type`: Optional model type, specified by `h2o_wave_ml.ModelType`.
- `kwargs`: Optional parameters to be passed to the model builder.

Returns:
    A Wave model.
    
    
### model.predict()
```python3
class Model:
    def predict(self, data: Optional[List[List]] = None, file_path: Optional[str] = None, **kwargs) -> List[Tuple]:
```

Returns the model's predictions for the given input rows. A file path or Python object can be passed.

- `data`: A list of rows of column values. First row has to contain the column headers.
- `file_path`: The file path to the dataset.

Example:
```python3
>>> from h2o_wave_ml import build_model
>>> model = build_model(...)
>>> # Three rows and two columns:
>>> model.predict([['ID', 'Letter'], [1, 'a'], [2, 'b'], [3, 'c']])
[(16.6,), (17.8,), (18.9,)]
```

Returns:
    A list of tuples representing predicted values.

### get_model()

```python3
def get_model(model_id: str, model_type: Optional[ModelType] = None) -> Model:
```

Retrieves a remote model using its ID.

- `model_id`: The unique ID of the model.
- `model_type`: (Optional) The type of the model, specified by `h2o_wave_ml.ModelType`.

Returns:
    The Wave model.

### save_model()

```python3
def save_model(model: Model, output_dir_path: str) -> str:
```

Saves a model to the given location.

- `model`: The model produced by `h2o_wave_ml.build_model`.
- `output_dir_path`: A directory where the model will be saved.

Returns:
    The file path to the saved model.

### load_model()

```python3
def load_model(file_path: str) -> Model:
```

Loads a saved model from the given location.

- `file_path`: Path to the saved model.

Returns:
    The Wave model.

## Environment variables

The environment variables ensure the correct behaviour of a function calls behind the scenes. Based on a setup the API might spawn a new `H2O-3` instance or use existing `DAI` instance, use plain password to authenticate or utilize OpenID Connect, etc.

Currently just one environment variable is available to set:

- `H2O_WAVE_ML_H2O3_URL`, if set the existing instance of `H2O-3` will be used instead of spawning a fresh one.

## Development setup 

A python of version `3.6.1` or greater is required.

1. Clone repo
2. Type `make setup`

## License

H2O Wave ML is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for more information.
