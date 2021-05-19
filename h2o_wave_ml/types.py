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

import abc
from enum import Enum
from typing import Optional, List, Tuple, Any

try:
    import pandas
    PandasDataFrame = pandas.DataFrame
except ModuleNotFoundError:
    PandasDataFrame = Any


class ModelMetric(Enum):
    """Determines a metric type."""

    AUTO = 1
    AUC = 2
    MSE = 3
    RMSE = 4
    MAE = 5
    RMSLE = 6
    DEVIANCE = 7
    LOGLOSS = 8
    AUCPR = 9
    LIFT_TOP_GROUP = 10
    MISCLASSIFICATION = 11
    MEAN_PER_CLASS_ERROR = 12


class ModelType(Enum):
    """Determines a type of the model backend."""

    H2O3 = 1
    DAI = 2


class TaskType(Enum):
    """Determines a machine learning task type."""

    CLASSIFICATION = 1
    REGRESSION = 2


class Model(abc.ABC):
    """Represents a predictive model."""

    def __init__(self, model_type: ModelType):
        self.type = model_type
        """A Wave model engine type."""

    @abc.abstractmethod
    def predict(self, data: Optional[List[List]] = None, file_path: str = '',
                test_df: Optional[PandasDataFrame] = None, **kwargs) -> List[Tuple]:
        """Returns the model's predictions for the given input rows.

        Args:
            data: A list of rows of column values. First row has to contain the column headers.
            file_path: The file path to the dataset.
            test_df: Pandas DataFrame.

        Returns:
            A list of tuples representing predicted values.

        Examples:
            >>> from h2o_wave_ml import build_model
            >>> model = build_model(...)
            >>> # Three rows and two columns:
            >>> model.predict([['ID', 'Letter'], [1, 'a'], [2, 'b'], [3, 'c']])
            [(16.6,), (17.8,), (18.9,)]

        """

    @property
    @abc.abstractmethod
    def endpoint_url(self) -> Optional[str]:
        """An endpoint url for a deployed model, if any."""
