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
from typing import Optional, Union, List, Tuple, Any

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


ModelEngineType = Enum('ModelEngineType', 'H2O3 DAI')
ModelMetric = Enum('ModelMetric', 'AUTO AUC MSE RMSE MAE RMSLE DEVIANCE LOGLOSS AUCPR LIFT_TOP_GROUP'
                   'MISCLASSIFICATION MEAN_PER_CLASS_ERROR')
DataSourceObj = Union[str, List[List]]


def _make_id() -> str:
    """Generates a random id."""

    return str(uuid.uuid4())


class _DataSource:
    """Helper class represents a various data sources that can be lazily transformed into another data type."""

    def __init__(self, data: DataSourceObj):
        self._data = data
        self._h2o3_frame: Optional[h2o.H2OFrame] = None

    def _to_h2o3_frame(self) -> h2o.H2OFrame:
        if isinstance(self._data, str):
            filepath = self._data
            if os.path.exists(filepath):
                return h2o.import_file(filepath)
            else:
                raise ValueError('file not found')
        elif isinstance(self._data, List):
            return h2o.H2OFrame(python_obj=self._data, header=1)
        raise ValueError('unknown data type')

    @property
    def h2o3_frame(self) -> h2o.H2OFrame:
        if self._h2o3_frame is None:
            self._h2o3_frame = self._to_h2o3_frame()
        return self._h2o3_frame

    @property
    def filepath(self) -> str:
        if isinstance(self._data, str):
            return self._data
        elif isinstance(self._data, List):
            raise NotImplementedError()
        raise ValueError('unknown data type')


class ModelEngine:
    """Represents a common interface for a model backend. It references DAI or H2O-3 under the hood."""

    def __init__(self, type_: ModelEngineType):
        self.type = type_
        """A Wave model engine type represented."""

    def predict(self, inputs: DataSourceObj, **kwargs) -> List[Tuple]:
        """Predicts values based on inputs.
        Args:
            inputs: A python obj or filename. A header needs to be specified for the python obj.
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


class _H2O3ModelEngine(ModelEngine):

    INIT = False
    MAX_RUNTIME_SECS = 60 * 60
    MAX_MODELS = 20
    INT_TO_CAT_THRESHOLD = 50

    def __init__(self, model: H2OEstimator):
        super().__init__(ModelEngineType.H2O3)
        self.model = model

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
    def build(cls, filepath: str, target_column: str, model_metric: ModelMetric, **aml_settings) -> ModelEngine:
        """Builds an H2O-3 based model and returns it in a `ModelEngine` wrapper."""

        cls._ensure()

        id_ = cls._make_project_id()
        aml = H2OAutoML(max_runtime_secs=aml_settings.get('max_runtime_secs', cls.MAX_RUNTIME_SECS),
                        max_models=aml_settings.get('max_models', cls.MAX_MODELS),
                        project_name=id_,
                        stopping_metric=model_metric.name,
                        sort_metric=model_metric.name)

        if os.path.exists(filepath):
            frame = h2o.import_file(filepath)
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
        return _H2O3ModelEngine(aml.leader)

    @classmethod
    def get(cls, id_: str) -> ModelEngine:
        """Gets a model identified by an AutoML project id.
        H2O-3 needs to be running standalone for this to work.

        Args:
            id_: Identification of the aml project on a running H2O-3 instance.
        Returns:
            A Wave model.
        """

        cls._ensure()

        aml = h2o.automl.get_automl(id_)
        return _H2O3ModelEngine(aml.leader)

    def predict(self, data: DataSourceObj, **kwargs) -> List[Tuple]:
        """Predicts on a model."""

        ds = _DataSource(data)
        input_frame = ds.h2o3_frame
        output_frame = self.model.predict(input_frame)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            filepath = os.path.join(tmp_dir_name, _make_id() + '.csv')
            h2o.download_csv(output_frame, filepath)
            prediction = dt.fread(filepath)
            return prediction.to_tuples()


def build_model(filepath: str, target_column: str, model_metric: ModelMetric = ModelMetric.AUTO,
                model_engine: Optional[ModelEngineType] = None, **kwargs) -> ModelEngine:
    """Trains a model.
    If `model_engine` not specified the function will determine correct backend based on a current environment.

    Args:
        filepath: The path to the training dataset.
        target_column: A name of the target column.
        model_metric: Optional evaluation metric to be used during modeling, specified by `h2o_wave_ml.ModelMetric`.
        model_engine: Optional modeling engine, specified by `h2o_wave_ml.ModelEngine`.
        kwargs: Optional parameters passed to the modeling engine.
    Returns:
        A Wave model.
    """

    if model_engine is not None:
        if model_engine == ModelEngineType.H2O3:
            return _H2O3ModelEngine.build(filepath, target_column, model_metric, **kwargs)
        raise NotImplementedError()

    return _H2O3ModelEngine.build(filepath, target_column, model_metric, **kwargs)


def get_model(id_: str, model_engine: Optional[ModelEngineType] = None) -> ModelEngine:
    """Gets a model that can be accessed on a backend.

    Args:
        id_: Identification of a model.
        model_engine: Optional modeling engine, specified by `h2o_wave_ml.ModelEngine`.
    Returns:
        A Wave model.
    """

    if model_engine is not None:
        if model_engine == ModelEngineType.H2O3:
            return _H2O3ModelEngine.get(id_)
        raise NotImplementedError()

    return _H2O3ModelEngine.get(id_)


def save_model(model_engine: ModelEngine, output_folder: str) -> str:
    """Saves a model to disk.

    Args:
       model_engine: A model engine produced by `h2o_wave_ml.build_model`.
       output_folder: A directory where the saved model will be put to.
    Returns:
        A path to a saved model.
    """

    if isinstance(model_engine, _H2O3ModelEngine):
        return h2o.download_model(model_engine.model, path=output_folder)
    raise NotImplementedError()


def load_model(filepath: str) -> ModelEngine:
    """Loads a model from disk into the instance.

    Args:
        filepath: Path to a saved model.
    Returns:
        A Wave model.
    """

    _H2O3ModelEngine._ensure()

    model = h2o.upload_model(path=filepath)
    return _H2O3ModelEngine(model)
