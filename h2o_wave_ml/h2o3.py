# Copyright 2021 H2O.ai, Inc.
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

from pathlib import Path
from typing import Optional, List, Tuple

import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.estimator_base import H2OEstimator

from .config import _config
from .types import Model, ModelType, ModelMetric, TaskType, PandasDataFrame
from .utils import _make_id, _remove_prefix


_INT_TO_CAT_THRESHOLD = 50


def _is_classification_task(frame: h2o.H2OFrame, target: str) -> bool:
    target_type = frame.type(target)
    if target_type == 'str':
        return True
    if target_type == 'int':
        uniques = frame[target].unique()
        if len(uniques) < _INT_TO_CAT_THRESHOLD:
            return True
    return False


def _create_h2o3_frame(data: Optional[List[List]] = None, file_path: str = '',
                       df: Optional[PandasDataFrame] = None) -> h2o.H2OFrame:
    if data is not None:
        return h2o.H2OFrame(python_obj=data, header=1)
    elif df is not None:
        return h2o.H2OFrame(python_obj=df)
    elif file_path:
        if Path(file_path).exists():
            return h2o.import_file(file_path)
        else:
            raise ValueError('file not found')
    raise ValueError('no data input')


def _make_project_id() -> str:
    """Generates a random project id."""

    # H2O-3 project name cannot start with a number (no matter it's string).
    u = _make_id()
    return f'aml-{u}'


def _decode_from_frame(data) -> List[Tuple]:
    ret = []
    for row in data:
        values = [float(item) for item in row[1:]]
        ret.append(tuple([row[0], *values]))
    return ret


class _H2O3Model(Model):

    _INIT = False
    _SUPPORTED_PARAMS = [
        '_h2o3_max_runtime_secs',
        '_h2o3_max_models',
        '_h2o3_nfolds',
        '_h2o3_balance_classes',
        '_h2o3_class_sampling_factors',
        '_h2o3_max_after_balance_size',
        '_h2o3_max_runtime_secs_per_model',
        '_h2o3_stopping_tolerance',
        '_h2o3_stopping_rounds',
        '_h2o3_seed',
        '_h2o3_exclude_algos',
        '_h2o3_include_algos',
        '_h2o3_modeling_plan',
        '_h2o3_preprocessing',
        '_h2o3_exploitation_ratio',
        '_h2o3_monotone_constraints',
        '_h2o3_keep_cross_validation_predictions',
        '_h2o3_keep_cross_validation_models',
        '_h2o3_keep_cross_validation_fold_assignment',
        '_h2o3_verbosity',
        '_h2o3_export_checkpoints_dir'
    ]

    def __init__(self, model: H2OEstimator):
        super().__init__(ModelType.H2O3)
        self.model = model

    @classmethod
    def ensure(cls):
        """Initializes H2O-3 library."""

        if not cls._INIT:
            if _config.h2o3_url:
                h2o.init(url=_config.h2o3_url)
            else:
                h2o.init()
            cls._INIT = True

    @classmethod
    def build(cls, train_file_path: str, train_df: Optional[PandasDataFrame], target_column: str,
              model_metric: ModelMetric, task_type: Optional[TaskType], categorical_columns: Optional[List[str]],
              feature_columns: Optional[List[str]], drop_columns: Optional[List[str]],
              validation_file_path: str, validation_df: Optional[PandasDataFrame], **kwargs) -> Model:
        """Builds an H2O-3 based model."""

        cls.ensure()

        id_ = _make_project_id()

        if train_file_path:
            if Path(train_file_path).exists():
                train_frame = h2o.import_file(train_file_path)
            else:
                raise ValueError('train file not found')
        elif train_df is not None:
            train_frame = h2o.H2OFrame(python_obj=train_df)
        else:
            raise ValueError('train data not supplied')

        if target_column not in train_frame.columns:
            raise ValueError('target column not found')

        if task_type is None:
            if _is_classification_task(train_frame, target_column):
                train_frame[target_column] = train_frame[target_column].asfactor()
        elif task_type == TaskType.CLASSIFICATION:
            train_frame[target_column] = train_frame[target_column].asfactor()

        if categorical_columns is not None:
            for column in categorical_columns:
                train_frame[column] = train_frame[column].ascharacter().asfactor()

        if feature_columns is not None:
            x = feature_columns
        elif drop_columns is not None:
            x = [col for col in train_frame.columns if col not in drop_columns]
        else:
            x = train_frame.columns

        if target_column in x:
            x.remove(target_column)

        params = {
            _remove_prefix(key, '_h2o3_'): kwargs[key]
            for key in kwargs
            if key in cls._SUPPORTED_PARAMS
        }

        aml = H2OAutoML(project_name=id_,
                        stopping_metric=model_metric.name,
                        sort_metric=model_metric.name,
                        **params)

        validation_frame = None
        if validation_file_path:
            if Path(validation_file_path).exists():
                validation_frame = h2o.import_file(validation_file_path)
            else:
                raise ValueError('validation file not found')
        elif validation_df is not None:
            validation_frame = h2o.H2OFrame(python_obj=validation_df)

        if validation_frame is not None and categorical_columns is not None:
            for column in categorical_columns:
                validation_frame[column] = validation_frame[column].ascharacter().asfactor()

        aml.train(x=x, y=target_column, training_frame=train_frame, validation_frame=validation_frame)

        if aml.leader is None:
            raise ValueError('no model available')

        return _H2O3Model(aml.leader)

    @classmethod
    def get(cls, model_id: str) -> Model:
        """Retrieves a remote model given its ID."""

        cls.ensure()

        aml = h2o.automl.get_automl(model_id)
        return _H2O3Model(aml.leader)

    def predict(self, data: Optional[List[List]] = None, file_path: str = '',
                test_df: Optional[PandasDataFrame] = None, **kwargs) -> List[Tuple]:
        input_frame = _create_h2o3_frame(data, file_path, test_df)
        output_frame = self.model.predict(input_frame)
        data = output_frame.as_data_frame(use_pandas=False, header=False)
        return _decode_from_frame(data)

    @property
    def endpoint_url(self) -> Optional[str]:
        return None
