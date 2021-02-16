# Copyright 2020 H2O.ai, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path
import uuid
from enum import Enum
import tempfile
from typing import Optional, List, Tuple, Any

import datatable as dt
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.estimator_base import H2OEstimator


def _get_env(key: str, default: Any):
    return os.environ.get(f'H2O_WAVE_ML_{key}', default)


class _Config:
    def __init__(self):
        self.h2o3_url = _get_env('H2O3_URL', '')


_config = _Config()


ModelType = Enum('ModelType', 'H2O3 DAI')
ModelMetric = Enum('ModelMetric', 'AUTO AUC MSE RMSE MAE RMSLE DEVIANCE LOGLOSS AUCPR LIFT_TOP_GROUP'
                   'MISCLASSIFICATION MEAN_PER_CLASS_ERROR')


def _make_id() -> str:
    """Generates a random id."""

    return str(uuid.uuid4())


class Model:
    """Represents a predictive model."""

    def __init__(self, type_: ModelType):
        self.type = type_
        """A Wave model engine type represented."""

    def predict(self, data: Optional[List[List]] = None, file_path: Optional[str] = None, **kwargs) -> List[Tuple]:
        """Returns the model's predictions for the given input rows.

        Args:
            data: A list of rows of column values. First row has to contain the column headers.
            file_path: The file path to the dataset.
        Returns:
            A list of tuples representing predicted values.
        Examples:
            >>> from h2o_wave_ml import build_model
            >>> model = build_model(...)
            >>> # Three rows and two columns:
            >>> model.predict([['ID', 'Letter'], [1, 'a'], [2, 'b'], [3, 'c']])
            [(16.6,), (17.8,), (18.9,)]
        """

        raise NotImplementedError()


class _H2O3Model(Model):

    INIT = False
    MAX_RUNTIME_SECS = 60 * 60
    MAX_MODELS = 20
    INT_TO_CAT_THRESHOLD = 50

    def __init__(self, model: H2OEstimator):
        super().__init__(ModelType.H2O3)
        self.model = model

    @staticmethod
    def _create_h2o3_frame(data: Optional[List[List]] = None, file_path: Optional[str] = None) -> h2o.H2OFrame:
        if data is not None:
            return h2o.H2OFrame(python_obj=data, header=1)
        elif file_path is not None:
            if os.path.exists(file_path):
                return h2o.import_file(file_path)
            else:
                raise ValueError('file not found')
        raise ValueError('no data input')

    @staticmethod
    def _make_project_id() -> str:
        """Generates a random project id."""

        # H2O-3 project name cannot start with a number (no matter it's string).
        u = _make_id()
        return f'aml-{u}'

    @classmethod
    def _ensure(cls):
        """Initializes H2O-3 library."""

        if not cls.INIT:
            if _config.h2o3_url != '':
                h2o.init(url=_config.h2o3_url)
            else:
                h2o.init()
            cls.INIT = True

    @classmethod
    def _is_classification_task(cls, frame: h2o.H2OFrame, target: str) -> bool:
        target_type = frame.type(target)
        if target_type == 'str':
            return True
        if target_type == 'int':
            uniques = frame[target].unique()
            if len(uniques) < cls.INT_TO_CAT_THRESHOLD:
                return True
        return False

    @classmethod
    def build(cls, file_path: str, target_column: str, model_metric: ModelMetric, **aml_settings) -> Model:
        """Builds an H2O-3 based model."""

        cls._ensure()

        id_ = cls._make_project_id()
        aml = H2OAutoML(max_runtime_secs=aml_settings.get('max_runtime_secs', cls.MAX_RUNTIME_SECS),
                        max_models=aml_settings.get('max_models', cls.MAX_MODELS),
                        project_name=id_,
                        stopping_metric=model_metric.name,
                        sort_metric=model_metric.name)

        if os.path.exists(file_path):
            frame = h2o.import_file(file_path)
        else:
            raise ValueError('file not found')

        cols = list(frame.columns)

        try:
            cols.remove(target_column)
        except ValueError:
            raise ValueError('no target column')

        if cls._is_classification_task(frame, target_column):
            frame[target_column] = frame[target_column].asfactor()

        aml.train(x=cols, y=target_column, training_frame=frame)
        return _H2O3Model(aml.leader)

    @classmethod
    def get(cls, model_id: str) -> Model:
        """Retrieves a remote model given its ID.
        By default, this refers to the ID of the H2O-3 AutoML object.

        Args:
            model_id: Identification of the aml project on a running H2O-3 instance.
        Returns:
            A Wave model.
        """

        cls._ensure()

        aml = h2o.automl.get_automl(model_id)
        return _H2O3Model(aml.leader)

    def predict(self, data: Optional[List[List]] = None, file_path: Optional[str] = None, **kwargs) -> List[Tuple]:
        """Returns the model's predictions for the given input rows."""

        input_frame = self._create_h2o3_frame(data, file_path)
        output_frame = self.model.predict(input_frame)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_file_path = os.path.join(tmp_dir_name, _make_id() + '.csv')
            h2o.download_csv(output_frame, tmp_file_path)
            prediction = dt.fread(tmp_file_path)
            return prediction.to_tuples()


def build_model(file_path: str, target_column: str, model_metric: ModelMetric = ModelMetric.AUTO,
                model_type: Optional[ModelType] = None, **kwargs) -> Model:
    """Trains a model.
    If `model_type` is not specified, it is inferred from the current environment. Defaults to a H2O-3 model.

    Args:
        file_path: The path to the training dataset.
        target_column: The name of the target column (the column to be predicted).
        model_metric: Optional evaluation metric to be used during modeling, specified by `h2o_wave_ml.ModelMetric`.
        model_type: Optional model type, specified by `h2o_wave_ml.ModelType`.
        kwargs: Optional parameters to be passed to the model builder.
    Returns:
        A Wave model.
    """

    if model_type is not None:
        if model_type == ModelType.H2O3:
            return _H2O3Model.build(file_path, target_column, model_metric, **kwargs)
        raise NotImplementedError()

    return _H2O3Model.build(file_path, target_column, model_metric, **kwargs)


def get_model(model_id: str, model_type: Optional[ModelType] = None) -> Model:
    """Retrieves a remote model using its ID.

    Args:
        model_id: The unique ID of the model.
        model_type: (Optional) The type of the model, specified by `h2o_wave_ml.ModelType`.
    Returns:
        The Wave model.
    """

    if model_type is not None:
        if model_type == ModelType.H2O3:
            return _H2O3Model.get(model_id)
        raise NotImplementedError()

    return _H2O3Model.get(model_id)


def save_model(model: Model, output_dir_path: str) -> str:
    """Saves a model to the given location.

    Args:
       model: The model produced by `h2o_wave_ml.build_model`.
       output_dir_path: A directory where the model will be saved.
    Returns:
        The file path to the saved model.
    """

    if isinstance(model, _H2O3Model):
        return h2o.download_model(model.model, path=output_dir_path)
    raise NotImplementedError()


def load_model(file_path: str) -> Model:
    """Loads a saved model from the given location.

    Args:
        file_path: Path to the saved model.
    Returns:
        The Wave model.
    """

    _H2O3Model._ensure()

    model = h2o.upload_model(path=file_path)
    return _H2O3Model(model)
