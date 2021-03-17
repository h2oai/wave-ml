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
from typing import Dict, Optional, List, Tuple, Any

import driverlessai
import datatable as dt
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.estimator_base import H2OEstimator


def _get_env(key: str, default: Any):
    return os.environ.get(f'H2O_WAVE_ML_{key}', default)


class _Config:
    def __init__(self):
        self.h2o3_url = _get_env('H2O3_URL', '')
        self.dai_address = _get_env('DAI_ADDRESS', '')
        self.dai_username = _get_env('DAI_USERNAME', '')
        self.dai_password = _get_env('DAI_PASSWORD', '')


_config = _Config()


ModelType = Enum('ModelType', 'H2O3 DAI')
ModelMetric = Enum('ModelMetric', 'AUTO AUC MSE RMSE MAE RMSLE DEVIANCE LOGLOSS AUCPR'
                                  'LIFT_TOP_GROUP MISCLASSIFICATION MEAN_PER_CLASS_ERROR')
TaskType = Enum('TaskType', 'CLASSIFICATION REGRESSION')


def _make_id() -> str:
    """Generates a random id."""

    return str(uuid.uuid4())


def _remove_prefix(text: str, prefix: str) -> str:
    return text[text.startswith(prefix) and len(prefix):]


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

    _INIT = False

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

        if not cls._INIT:
            if _config.h2o3_url != '':
                h2o.init(url=_config.h2o3_url)
            else:
                h2o.init()
            cls._INIT = True

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
    def build(cls, file_path: str, target_column: str, model_metric: ModelMetric, task_type: Optional[TaskType],
              **kwargs) -> Model:
        """Builds an H2O-3 based model."""

        cls._ensure()

        id_ = cls._make_project_id()
        aml = H2OAutoML(max_runtime_secs=kwargs.get('_h2o3_max_runtime_secs', cls.MAX_RUNTIME_SECS),
                        max_models=kwargs.get('_h2o3_max_models', cls.MAX_MODELS),
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

        if task_type is None:
            if cls._is_classification_task(frame, target_column):
                frame[target_column] = frame[target_column].asfactor()
        elif task_type == TaskType.CLASSIFICATION:
            frame[target_column] = frame[target_column].asfactor()

        aml.train(x=cols, y=target_column, training_frame=frame)
        return _H2O3Model(aml.leader)

    @classmethod
    def get(cls, model_id: str) -> Model:
        """Retrieves a remote model given its ID."""

        cls._ensure()

        aml = h2o.automl.get_automl(model_id)
        return _H2O3Model(aml.leader)

    def predict(self, data: Optional[List[List]] = None, file_path: Optional[str] = None, **kwargs) -> List[Tuple]:
        input_frame = self._create_h2o3_frame(data, file_path)
        output_frame = self.model.predict(input_frame)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_file_path = os.path.join(tmp_dir_name, _make_id() + '.csv')
            h2o.download_csv(output_frame, tmp_file_path)
            prediction = dt.fread(tmp_file_path)
            return prediction.to_tuples()


class _DAIModel(Model):

    _INSTANCE = None

    INT_TO_CAT_THRESHOLD = 50

    def __init__(self, experiment):
        super().__init__(ModelType.DAI)
        self.experiment = experiment

    @classmethod
    def _get_instance(cls):
        if cls._INSTANCE is None:
            if _config.dai_address:
                cls._INSTANCE = driverlessai.Client(address=_config.dai_address,
                                                    username=_config.dai_username,
                                                    password=_config.dai_password)
            else:
                raise RuntimeError('no backend service available')
        return cls._INSTANCE

    @classmethod
    def _determine_task_type(cls, summary) -> str:
        if summary.data_type in ('int', 'real'):
            if summary.unique > cls.INT_TO_CAT_THRESHOLD:
                return 'regression'
        return 'classification'

    @staticmethod
    def _transpose(data: Optional[List[List]] = None) -> Dict[str, List]:
        return {
            name: [data[row_i][col_i] for row_i in range(1, len(data))]
            for col_i, name in enumerate(data[0])
        }

    @staticmethod
    def _extract_class(name: str) -> str:
        """Extract a predicted class name from DAI column name.

        Examples:
            >>> _DAIModel._extract_class('target_column.class1')
            'class1'
        """

        return name.split('.')[-1]

    @classmethod
    def _get_prediction_output(cls, df: dt.frame) -> List[Tuple]:
        names = [cls._extract_class(name) for name in df.names]

        ret = []
        for i in range(df.nrows):
            row = df[i, :].to_tuples()[0]
            index = row.index(max(row))
            ret.append(tuple([names[index], *row]))

        return ret

    @classmethod
    def build(cls, file_path: str, target_column: str, model_metric: ModelMetric, task_type: Optional[TaskType],
              **kwargs) -> Model:
        """Builds DAI based model."""

        dai = cls._get_instance()

        dataset_id = _make_id()
        dataset = dai.datasets.create(file_path, name=dataset_id)

        try:
            summary = dataset.column_summaries(columns=[target_column])[0]
        except KeyError:
            raise ValueError('no target column')

        params = {
            _remove_prefix(key, '_dai_'): kwargs[key]
            for key in kwargs
            if key in ('_dai_accuracy', '_dai_time', '_dai_interpretability')
        }

        if task_type is None:
            task = cls._determine_task_type(summary)
        else:
            task = task_type.name.lower()

        if model_metric != ModelMetric.AUTO:
            params['scorer'] = model_metric.name

        ex = dai.experiments.create(
            train_dataset=dataset,
            target_column=target_column,
            task=kwargs.get('_dai_task', task),
            **params,
        )

        return _DAIModel(ex)

    @classmethod
    def get(cls, experiment_id: str) -> Model:
        """Retrieves a remote model given its ID."""

        dai = cls._get_instance()
        return _DAIModel(dai.experiments.get(experiment_id))

    def predict(self, data: Optional[List[List]] = None, file_path: Optional[str] = None, **_kwargs) -> List[Tuple]:
        dai = self._get_instance()
        dataset_id = _make_id()

        if data is not None:
            df = dt.Frame(self._transpose(data))
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                tmp_file_path = os.path.join(tmp_dir_name, _make_id() + '.csv')
                df.to_csv(tmp_file_path, header=True)
                dataset = dai.datasets.create(tmp_file_path, name=dataset_id)
        elif file_path is not None:
            dataset = dai.datasets.create(file_path, name=dataset_id)
        else:
            raise ValueError('no data input')

        prediction_obj = self.experiment.predict(dataset=dataset)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_file_path = os.path.join(tmp_dir_name, _make_id() + '.csv')
            prediction_obj.download(dst_file=tmp_file_path)
            prediction = dt.fread(tmp_file_path)
            return self._get_prediction_output(prediction)


def build_model(file_path: str, target_column: str, model_metric: ModelMetric = ModelMetric.AUTO,
                task_type: Optional[TaskType] = None, model_type: Optional[ModelType] = None, **kwargs) -> Model:
    """Trains a model.
    If `model_type` is not specified, it is inferred from the current environment. Defaults to a H2O-3 model.

    Args:
        file_path: The path to the training dataset.
        target_column: The name of the target column (the column to be predicted).
        model_metric: Optional evaluation metric to be used during modeling, specified by `h2o_wave_ml.ModelMetric`.
        task_type: Optional task type, specified by `h2o_wave_ml.TaskType`.
        model_type: Optional model type, specified by `h2o_wave_ml.ModelType`.
        kwargs: Optional parameters to be passed to the model builder.
    Returns:
        A Wave model.
    """

    if model_type is not None:
        if model_type == ModelType.H2O3:
            return _H2O3Model.build(file_path, target_column, model_metric, task_type, **kwargs)
        elif model_type == ModelType.DAI:
            return _DAIModel.build(file_path, target_column, model_metric, task_type, **kwargs)

    if _config.dai_address:
        return _DAIModel.build(file_path, target_column, model_metric, task_type, **kwargs)

    return _H2O3Model.build(file_path, target_column, model_metric, task_type, **kwargs)


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
        elif model_type == ModelType.DAI:
            return _DAIModel.get(model_id)

    if _config.dai_address:
        return _DAIModel.get(model_id)

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
