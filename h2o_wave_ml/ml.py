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

from typing import Optional, List

import h2o

from .config import _config
from .dai import _DAIModel
from .h2o3 import _H2O3Model
from .types import Model, ModelMetric, ModelType, TaskType, PandasDataFrame


def build_model(*, target_column: str, train_file_path: str = '', train_df: Optional[PandasDataFrame] = None,
                model_metric: ModelMetric = ModelMetric.AUTO, task_type: Optional[TaskType] = None,
                model_type: Optional[ModelType] = None, categorical_columns: Optional[List[str]] = None,
                feature_columns: Optional[List[str]] = None, drop_columns: Optional[List[str]] = None,
                validation_file_path: str = '', validation_df: Optional[PandasDataFrame] = None,
                access_token: str = '', refresh_token: str = '', **kwargs) -> Model:
    """Trains a model.

    The function has to be called with `target_column` and `train_file_path` or `train_df` at least to be functionable.
    If `model_type` is not specified, it is inferred from the current environment. Defaults to an H2O-3 model.

    Args:
        target_column: The name of the target column (the column to be predicted).
        train_file_path: The path to the training dataset.
        train_df: Pandas DataFrame as a training set instead of file.
        model_metric: Optional evaluation metric to be used for modeling.
        task_type: Optional task type. Will be automatically determined if it's not specified.
        model_type: Optional model type.
        categorical_columns: Optional list of column names to be converted (from numeric) to categorical.
        feature_columns: Optional list of column names to be used for modeling.
        drop_columns: Optional list of column names to be dropped before modeling.
        validation_file_path: Optional path to a validation dataset.
        validation_df: Optional Pandas DataFrame as a validation dataset.
        access_token: Optional access token if engine needs to be authenticated.
        refresh_token: Optional refresh token if model needs to be authenticated.
        kwargs: Optional parameters to be passed to the model builder, Steam or MLOps.

    Kwargs:

        The list of the supported **DAI** parameters. The parameters description can be found [here](http://docs.h2o.ai/driverless-ai/pyclient/docs/html/client.html).

        *_dai_accuracy*<br />
        *_dai_time*<br />
        *_dai_interpretability*<br />
        *_dai_models*<br />
        *_dai_transformers*<br />
        *_dai_weight_column*<br />
        *_dai_fold_column*<br />
        *_dai_time_column*<br />
        *_dai_time_groups_columns*<br />
        *_dai_unavailable_at_prediction_time_columns*<br />
        *_dai_enable_gpus*<br />
        *_dai_reproducible*<br />
        *_dai_time_period_in_seconds*<br />
        *_dai_num_prediction_periods*<br />
        *_dai_num_gap_periods*<br />
        *_dai_config_overrides*<br />

        The list of the supported **H2O-3** parameters. The parameters description can be found [here](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html#required-parameters).

        [*_h2o3_max_runtime_secs*](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/max_runtime_secs.html)<br />
        [*_h2o3_max_models*](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/max_models.html)<br />
        [*_h2o3_nfolds*](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/nfolds.html)<br />
        [*_h2o3_balance_classes*](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/balance_classes.html)<br />
        [*_h2o3_class_sampling_factors*](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/class_sampling_factors.html)<br />
        [*_h2o3_max_after_balance_size*](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/max_after_balance_size.html)<br />
        [*_h2o3_max_runtime_secs_per_model*](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/max_runtime_secs_per_model.html)<br />
        [*_h2o3_stopping_tolerance*](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/stopping_tolerance.html)<br />
        [*_h2o3_stopping_rounds*](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/stopping_rounds.html)<br />
        [*_h2o3_seed*](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/seed.html)<br />
        [*_h2o3_exclude_algos*](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/exclude_algos.html)<br />
        [*_h2o3_include_algos*](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/include_algos.html)<br />
        *_h2o3_modeling_plan*<br />
        *_h2o3_preprocessing*<br />
        *_h2o3_exploitation_ratio*<br />
        [*_h2o3_monotone_constraints*](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/monotone_constraints.html)<br />
        [*_h2o3_keep_cross_validation_predictions*](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/keep_cross_validation_predictions.html)<br />
        [*_h2o3_keep_cross_validation_models*](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/keep_cross_validation_models.html)<br />
        [*_h2o3_keep_cross_validation_fold_assignment*](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/keep_cross_validation_fold_assignment.html)<br />
        *_h2o3_verbosity*<br />
        [*_h2o3_export_checkpoints_dir*](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/export_checkpoints_dir.html)<br />

        The list of the supported **Steam** options.

        *_steam_dai_instance_name*<br />
        *_steam_dai_multinode_name*<br />

        The list of the supported **MLOps** options.

        *_mlops_deployment_env*<br />

    Returns:
        The Wave model.

    """

    if feature_columns is not None and drop_columns is not None:
        raise ValueError('expected either `feature_columns` or `drop_columns` args')

    if model_type is not None:
        if model_type == ModelType.H2O3:
            return _H2O3Model.build(train_file_path, train_df, target_column, model_metric, task_type,
                                    categorical_columns, feature_columns, drop_columns, validation_file_path,
                                    validation_df, **kwargs)
        elif model_type == ModelType.DAI:
            return _DAIModel.build(train_file_path, train_df, target_column, model_metric, task_type,
                                   categorical_columns, feature_columns, drop_columns, validation_file_path,
                                   validation_df, access_token, refresh_token, **kwargs)

    if _config.dai_address or _config.steam_address:
        return _DAIModel.build(train_file_path, train_df, target_column, model_metric, task_type, categorical_columns,
                               feature_columns, drop_columns, validation_file_path, validation_df, access_token,
                               refresh_token, **kwargs)

    return _H2O3Model.build(train_file_path, train_df, target_column, model_metric, task_type, categorical_columns,
                            feature_columns, drop_columns, validation_file_path, validation_df, **kwargs)


def get_model(model_id: str = '', endpoint_url: str = '', model_type: Optional[ModelType] = None,
              access_token: str = '', refresh_token: str = '') -> Optional[Model]:
    """Retrieves a remote model using its ID or url.

    Args:
        model_id: The unique ID of the model.
        endpoint_url: The endpoint url for deployed model.
        model_type: Optional type of the model.
        access_token: Optional access token if model needs to be authenticated.
        refresh_token: Optional refresh token if model needs to be authenticated.

    Returns:
        The Wave model.

    """

    if model_type is not None:
        if model_type == ModelType.H2O3:
            return _H2O3Model.get(model_id)
        elif model_type == ModelType.DAI:
            return _DAIModel.get(model_id, endpoint_url, access_token, refresh_token)

    if _config.dai_address or _config.steam_address:
        return _DAIModel.get(model_id, endpoint_url, access_token, refresh_token)

    return _H2O3Model.get(model_id)


def save_model(model: Model, *, output_dir_path: str) -> str:
    """Saves a model to the given location.

    Args:
       model: The model to store.
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

    _H2O3Model.ensure()

    model = h2o.upload_model(path=file_path)
    return _H2O3Model(model)
