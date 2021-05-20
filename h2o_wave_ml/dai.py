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

import csv
from pathlib import Path
import time
from typing import Dict, Optional, List, Tuple, IO

import driverlessai
try:
    from h2osteam.clients import DriverlessClient, MultinodeClient
    import mlops
    from _mlops.deployer import exceptions as ml_excp
except ModuleNotFoundError:
    pass
import requests

from .config import _config
from .types import Model, ModelType, ModelMetric, TaskType, PandasDataFrame
from .utils import _make_id, _remove_prefix, _is_mlops_imported, _connect_to_steam, _refresh_token


_INT_TO_CAT_THRESHOLD = 50
_MLOPS_REFRESH_STATUS_INTERVAL = 1
_MLOPS_MAX_WAIT_TIME = 300


def _determine_task_type(summary) -> str:
    if summary.data_type in ('int', 'real'):
        if summary.unique > _INT_TO_CAT_THRESHOLD:
            return 'regression'
    return 'classification'


def _make_project_id() -> str:
    """Generates a random project id."""

    u = _make_id()
    return f'wave-ml-{u}'


def _wait_for_deployment(mlops_client, deployment_id: str):

    deadline = time.monotonic() + _MLOPS_MAX_WAIT_TIME

    status = mlops_client.deployer.deployment_status.get_deployment_status(
        mlops.DeployGetDeploymentStatusRequest(deployment_id=deployment_id)).deployment_status
    while status.state != mlops.DeployDeploymentState.HEALTHY:
        time.sleep(_MLOPS_REFRESH_STATUS_INTERVAL)
        status = mlops_client.deployer.deployment_status.get_deployment_status(
            mlops.DeployGetDeploymentStatusRequest(deployment_id=deployment_id)).deployment_status
        if time.monotonic() > deadline:
            raise RuntimeError('deployment timeout error')


def _encode_from_data(data: List[List]) -> Dict:
    if len(data) < 2:
        raise ValueError('invalid input format')

    return {
        'fields': data[0],
        'rows':  [[str(item) for item in row] for row in data[1:]],
    }


def _encode_from_csv(csvfile: IO) -> Dict:
    reader = csv.reader(csvfile)
    return {
        'fields': next(reader),
        'rows': [row for row in reader],
    }


def _encode_from_pandas(df: PandasDataFrame) -> Dict:
    return {
        'fields': list(df.columns),
        'rows': [[str(item) for item in row] for _i, row in df.iterrows()]
    }


def _extract_class(name: str) -> str:
    """Extract a predicted class name from DAI column name.

    Examples:
        >>> _extract_class('target_column.class1')
        'class1'

    """

    return name.split('.')[-1]


def _decode_from_deployment(data) -> List[Tuple]:
    names = [_extract_class(name) for name in data['fields']]

    ret = []
    for row in data['score']:
        values = [float(item) for item in row]
        index = values.index(max(values))
        ret.append(tuple([names[index], *values]))

    return ret


class _DAIModel(Model):

    _INSTANCE = None
    _SUPPORTED_PARAMS = [
        '_dai_accuracy',
        '_dai_time',
        '_dai_interpretability',
        '_dai_models',
        '_dai_transformers',
        '_dai_weight_column',
        '_dai_fold_column',
        '_dai_time_column',
        '_dai_time_groups_columns',
        '_dai_unavailable_at_prediction_time_columns',
        '_dai_enable_gpus',
        '_dai_reproducible',
        '_dai_time_period_in_seconds',
        '_dai_num_prediction_periods',
        '_dai_num_gap_periods',
        '_dai_config_overrides'
    ]

    def __init__(self, endpoint_url: str):
        super().__init__(ModelType.DAI)
        self._endpoint_url = endpoint_url

    @classmethod
    def _get_instance(cls, access_token: str = '', **kwargs):

        instance_name = kwargs.get('_steam_dai_instance_name', _config.steam_instance_name)
        multinode_name = kwargs.get('_steam_dai_multinode_name', _config.steam_cluster_name)

        if cls._INSTANCE is None:
            if _config.dai_address:
                cls._INSTANCE = driverlessai.Client(address=_config.dai_address,
                                                    username=_config.dai_username,
                                                    password=_config.dai_password)
            elif _config.steam_address:

                _connect_to_steam(access_token)

                if multinode_name:
                    instance = MultinodeClient.get_cluster(name=multinode_name)
                    if not instance.is_master_ready():
                        raise RuntimeError('DAI master node not ready')
                elif instance_name:
                    instance = DriverlessClient.get_instance(name=instance_name)
                    if instance.status() == 'stopped':
                        raise RuntimeError('DAI instance not ready: stopped')
                    elif instance.status() == 'failed':
                        raise RuntimeError('DAI instance not ready: failed')
                else:
                    raise RuntimeError('no DAI resource specified')

                cls._INSTANCE = instance.connect()
            else:
                raise RuntimeError('no backend service available')
        return cls._INSTANCE

    @classmethod
    def _build_model(cls, train_file_path: str, train_df: Optional[PandasDataFrame], target_column: str,
                     model_metric: ModelMetric, task_type: Optional[TaskType], categorical_columns: Optional[List[str]],
                     feature_columns: Optional[List[str]], drop_columns: Optional[List[str]],
                     validation_file_path: str, validation_df: Optional[PandasDataFrame], access_token: str, **kwargs):

        dai = cls._get_instance(access_token, **kwargs)

        train_dataset_id = _make_id()

        if train_file_path:
            if Path(train_file_path).exists():
                train_dataset = dai.datasets.create(data=train_file_path, name=train_dataset_id)
            else:
                raise ValueError('train file not found')
        elif train_df is not None:
            train_dataset = dai.datasets.create(data=train_df, name=train_dataset_id)
        else:
            raise ValueError('train data not supplied')

        try:
            train_summary = train_dataset.column_summaries(columns=[target_column])[0]
        except KeyError:
            raise ValueError('target column not found')

        if task_type is None:
            task = _determine_task_type(train_summary)
        else:
            task = task_type.name.lower()

        column_types = {}
        if categorical_columns is not None:
            for column in categorical_columns:
                column_types[column] = 'categorical'
            train_dataset.set_logical_types(column_types)

        params = {
            _remove_prefix(key, '_dai_'): kwargs[key]
            for key in kwargs
            if key in cls._SUPPORTED_PARAMS
        }

        if model_metric != ModelMetric.AUTO:
            params['scorer'] = model_metric.name

        validation_dataset = None
        if validation_file_path:
            if Path(validation_file_path).exists():
                validation_dataset = dai.datasets.create(data=validation_file_path, name=_make_id())
            else:
                raise ValueError('validation file not found')
        elif validation_df is not None:
            validation_dataset = dai.datasets.create(data=validation_df, name=_make_id())

        if validation_dataset is not None:
            if categorical_columns is not None:
                validation_dataset.set_logical_types(column_types)
            params['validation_dataset'] = validation_dataset

        if feature_columns is not None:
            params['drop_columns'] = [col for col in train_dataset.columns if col not in feature_columns]
        elif drop_columns is not None:
            params['drop_columns'] = drop_columns

        if target_column in params.get('drop_columns', []):
            params['drop_columns'].remove(target_column)

        experiment = dai.experiments.create(
            train_dataset=train_dataset,
            target_column=target_column,
            task=task,
            **params,
        )

        return experiment

    @classmethod
    def _deploy_model(cls, experiment, access_token: str, deployment_env: str) -> str:

        if not _is_mlops_imported():
            raise RuntimeError('no MLOps package installed (install mlops)')

        if not _config.mlops_gateway:
            raise ValueError('no MLOps gateway specified')

        dai = cls._INSTANCE  # This instance should already by present.
        prj = dai.projects.create(name=_make_project_id())
        prj.link_experiment(experiment)

        mlops_client = mlops.Client(gateway_url=_config.mlops_gateway,
                                    token_provider=lambda: access_token)

        # Fetch available deployment environments.
        deployment_envs = mlops_client.storage.deployment_environment.list_deployment_environments(
            mlops.StorageListDeploymentEnvironmentsRequest(prj.key))

        # Look for the ID of the selected deployment environment
        for env in deployment_envs.deployment_environment:
            if env.display_name == deployment_env:
                deployment_env_id = env.id
                break
        else:
            raise ValueError('unknown deployment environment')

        deployment = mlops_client.storage.deployment_environment.deploy(
            mlops.StorageDeployRequest(
                experiment_id=experiment.key,
                deployment_environment_id=deployment_env_id,
                type=mlops.StorageDeploymentType.SINGLE_MODEL,
                metadata=mlops.StorageMetadata(
                    values={"deploy/authentication/enabled": mlops.StorageValue(bool_value=False)}
                ),
            )
        )

        project_id = deployment.deployment.project_id
        deployment_id = deployment.deployment.id

        _wait_for_deployment(mlops_client, deployment_id)

        statuses = mlops_client.deployer.deployment_status.list_deployment_statuses(
            mlops.DeployListDeploymentStatusesRequest(project_id=project_id))
        return statuses.deployment_status[0].scorer.score.url

    @classmethod
    def build(cls, train_file_path: str, train_df: Optional[PandasDataFrame], target_column: str,
              model_metric: ModelMetric, task_type: Optional[TaskType], categorical_columns: Optional[List[str]],
              feature_columns: Optional[List[str]], drop_columns: Optional[List[str]],
              validation_file_path: str, validation_df: Optional[PandasDataFrame],
              access_token: str, refresh_token: str, **kwargs) -> Model:
        """Builds DAI based model."""

        if refresh_token:
            access_token, refresh_token = _refresh_token(refresh_token, _config.oidc_provider_url,
                                                         _config.oidc_client_id, _config.oidc_client_secret)

        experiment = cls._build_model(train_file_path=train_file_path, train_df=train_df, target_column=target_column,
                                      model_metric=model_metric, task_type=task_type,
                                      categorical_columns=categorical_columns, feature_columns=feature_columns,
                                      drop_columns=drop_columns, validation_file_path=validation_file_path,
                                      validation_df=validation_df, access_token=access_token, **kwargs)

        if refresh_token:
            access_token, refresh_token = _refresh_token(refresh_token, _config.oidc_provider_url,
                                                         _config.oidc_client_id, _config.oidc_client_secret)
        elif not access_token and not refresh_token:
            raise ValueError('no token credentials for MLOps')

        deployment_env = kwargs.get('_mlops_deployment_env', 'PROD')
        endpoint_url = cls._deploy_model(experiment, access_token, deployment_env)

        return _DAIModel(endpoint_url)

    @classmethod
    def get(cls, project_id: str, endpoint_url: str = '', access_token: str = '',
            refresh_token: str = '') -> Optional[Model]:
        """Retrieves a remote model given its ID."""

        if endpoint_url:
            return _DAIModel(endpoint_url)

        if not _is_mlops_imported():
            raise RuntimeError('no MLOps package installed (install mlops)')

        if not _config.mlops_gateway:
            raise ValueError('no MLOps gateway specified')

        if refresh_token:
            access_token, _ = _refresh_token(refresh_token, _config.oidc_provider_url,
                                             _config.oidc_client_id, _config.oidc_client_secret)

        mlops_client = mlops.Client(gateway_url=_config.mlops_gateway,
                                    token_provider=lambda: access_token)

        try:
            statuses = mlops_client.deployer.deployment_status.list_deployment_statuses(
                mlops.DeployListDeploymentStatusesRequest(project_id=project_id))
        except ml_excp.ApiException:
            return None

        # There should be a strategy to pick the right deployment instead of picking a zeroth one.
        endpoint_url = statuses.deployment_status[0].scorer.score.url

        return _DAIModel(endpoint_url)

    def predict(self, data: Optional[List[List]] = None, file_path: str = '',
                test_df: Optional[PandasDataFrame] = None, **kwargs) -> List[Tuple]:

        if data is not None:
            payload = _encode_from_data(data)
        elif test_df is not None:
            payload = _encode_from_pandas(test_df)
        elif file_path:
            with open(file_path) as csvfile:
                payload = _encode_from_csv(csvfile)
        else:
            raise ValueError('no data input')

        r = requests.post(self._endpoint_url, json=payload)
        r.raise_for_status()
        return _decode_from_deployment(r.json())

    @property
    def endpoint_url(self) -> Optional[str]:
        return self._endpoint_url
