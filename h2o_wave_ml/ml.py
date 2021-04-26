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

import abc
import csv
import os.path
import uuid
from enum import Enum
import sys
import time
from typing import Dict, Optional, List, Tuple, Any, Union, IO
from urllib.parse import urljoin

import driverlessai
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.estimator_base import H2OEstimator
import requests

try:
    import h2osteam
    from h2osteam.clients import DriverlessClient, MultinodeClient
    import mlops
    from _mlops.deployer import exceptions as ml_excp
except ModuleNotFoundError:
    pass


def _get_env(key: str, default: Any = ''):
    return os.environ.get(f'H2O_WAVE_{key}', default)


def _is_enabled(value: Union[bool, str]):
    if isinstance(value, bool):
        return value
    if value.lower() == 'true':
        return True
    return False


class _Config:
    def __init__(self):

        # Wave ML namespace.
        self.h2o3_url = _get_env('ML_H2O3_URL')
        self.dai_address = _get_env('ML_DAI_ADDRESS')
        self.dai_username = _get_env('ML_DAI_USERNAME')
        self.dai_password = _get_env('ML_DAI_PASSWORD')
        self.steam_address = _get_env('ML_STEAM_ADDRESS')
        self.steam_refresh_token = _get_env('ML_STEAM_REFRESH_TOKEN')
        self.steam_instance_name = _get_env('ML_STEAM_INSTANCE_NAME')
        self.steam_cluster_name = _get_env('ML_STEAM_CLUSTER_NAME')
        self.steam_verify_ssl = _is_enabled(_get_env('ML_STEAM_VERIFY_SSL', True))
        self.mlops_gateway = _get_env('ML_MLOPS_GATEWAY')

        # OIDC namespace.
        self.oidc_provider_url = _get_env('OIDC_PROVIDER_URL')
        self.oidc_client_id = _get_env('OIDC_CLIENT_ID')
        self.oidc_client_secret = _get_env('OIDC_CLIENT_SECRET')


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


def _is_package_imported(name: str) -> bool:
    try:
        sys.modules[name]
    except KeyError:
        return False
    return True


def _is_steam_imported() -> bool:
    return _is_package_imported('h2osteam')


def _is_mlops_imported() -> bool:
    return _is_package_imported('mlops')


class Model(abc.ABC):
    """Represents a predictive model."""

    def __init__(self, model_type: ModelType):
        self.type = model_type
        """A Wave model engine type."""

    @abc.abstractmethod
    def predict(self, data: Optional[List[List]] = None, file_path: str = '', **kwargs) -> List[Tuple]:
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

    @property
    @abc.abstractmethod
    def endpoint_url(self) -> Optional[str]:
        """An endpoint url for a deployed model, if any."""


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

    INT_TO_CAT_THRESHOLD = 50

    def __init__(self, model: H2OEstimator):
        super().__init__(ModelType.H2O3)
        self.model = model

    @staticmethod
    def _create_h2o3_frame(data: Optional[List[List]] = None, file_path: str = '') -> h2o.H2OFrame:
        if data is not None:
            return h2o.H2OFrame(python_obj=data, header=1)
        elif file_path:
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
            if _config.h2o3_url:
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

        params = {
            _remove_prefix(key, '_h2o3_'): kwargs[key]
            for key in kwargs
            if key in cls._SUPPORTED_PARAMS
        }

        aml = H2OAutoML(project_name=id_,
                        stopping_metric=model_metric.name,
                        sort_metric=model_metric.name,
                        **params)

        if os.path.exists(file_path):
            frame = h2o.import_file(file_path)
        else:
            raise ValueError('file not found')

        if target_column not in frame.columns:
            raise ValueError('target column not found')

        if task_type is None:
            if cls._is_classification_task(frame, target_column):
                frame[target_column] = frame[target_column].asfactor()
        elif task_type == TaskType.CLASSIFICATION:
            frame[target_column] = frame[target_column].asfactor()

        aml.train(y=target_column, training_frame=frame)
        return _H2O3Model(aml.leader)

    @classmethod
    def get(cls, model_id: str) -> Model:
        """Retrieves a remote model given its ID."""

        cls._ensure()

        aml = h2o.automl.get_automl(model_id)
        return _H2O3Model(aml.leader)

    @staticmethod
    def _decode_from_frame(data) -> List[Tuple]:
        ret = []
        for row in data:
            values = [float(item) for item in row[1:]]
            ret.append(tuple([row[0], *values]))
        return ret

    def predict(self, data: Optional[List[List]] = None, file_path: str = '', **kwargs) -> List[Tuple]:
        input_frame = self._create_h2o3_frame(data, file_path)
        output_frame = self.model.predict(input_frame)
        data = output_frame.as_data_frame(use_pandas=False, header=False)
        return self._decode_from_frame(data)

    @property
    def endpoint_url(self) -> Optional[str]:
        return None


class _DAIModel(Model):

    _INSTANCE = None

    INT_TO_CAT_THRESHOLD = 50
    MLOPS_REFRESH_STATUS_INTERVAL = 1
    MLOPS_MAX_WAIT_TIME = 300

    def __init__(self, endpoint_url: str):
        super().__init__(ModelType.DAI)
        self._endpoint_url = endpoint_url

    @staticmethod
    def _make_project_id() -> str:
        """Generates a random project id."""

        u = _make_id()
        return f'wave-ml-{u}'

    @classmethod
    def _get_instance(cls, access_token: str = ''):
        if cls._INSTANCE is None:
            if _config.dai_address:
                cls._INSTANCE = driverlessai.Client(address=_config.dai_address,
                                                    username=_config.dai_username,
                                                    password=_config.dai_password)
            elif _config.steam_address:

                if not _is_steam_imported():
                    raise RuntimeError('no Steam package installed (install h2osteam)')

                if _config.steam_refresh_token:
                    h2osteam.login(url=_config.steam_address, refresh_token=_config.steam_refresh_token,
                                   verify_ssl=_config.steam_verify_ssl)
                elif access_token:
                    h2osteam.login(url=_config.steam_address, access_token=access_token,
                                   verify_ssl=_config.steam_verify_ssl)
                else:
                    raise RuntimeError('no Steam credentials')

                if _config.steam_cluster_name:
                    instance = MultinodeClient.get_cluster(name=_config.steam_cluster_name)
                    if not instance.is_master_ready():
                        raise RuntimeError('DAI master node not ready')
                elif _config.steam_instance_name:
                    instance = DriverlessClient.get_instance(name=_config.steam_instance_name)
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
    def _determine_task_type(cls, summary) -> str:
        if summary.data_type in ('int', 'real'):
            if summary.unique > cls.INT_TO_CAT_THRESHOLD:
                return 'regression'
        return 'classification'

    @staticmethod
    def _refresh_token(refresh_token: str, provider_url: str, client_id: str, client_secret: str) -> Tuple[str, str]:

        provider_url = f'{provider_url}/' if not provider_url.endswith('/') else provider_url
        r = requests.get(urljoin(provider_url, '.well-known/openid-configuration'))
        r.raise_for_status()
        conf_data = r.json()

        token_endpoint_url = conf_data['token_endpoint']

        payload = dict(
            client_id=client_id,
            client_secret=client_secret,
            grant_type='refresh_token',
            refresh_token=refresh_token,
        )
        r = requests.post(token_endpoint_url, data=payload)
        r.raise_for_status()
        token_data = r.json()
        return token_data['access_token'], token_data['refresh_token']

    @classmethod
    def _wait_for_deployment(cls, mlops_client, deployment_id: str):

        deadline = time.monotonic() + cls.MLOPS_MAX_WAIT_TIME

        status = mlops_client.deployer.deployment_status.get_deployment_status(
            mlops.DeployGetDeploymentStatusRequest(deployment_id=deployment_id)).deployment_status
        while status.state != mlops.DeployDeploymentState.HEALTHY:
            time.sleep(cls.MLOPS_REFRESH_STATUS_INTERVAL)
            status = mlops_client.deployer.deployment_status.get_deployment_status(
                mlops.DeployGetDeploymentStatusRequest(deployment_id=deployment_id)).deployment_status
            if time.monotonic() > deadline:
                raise RuntimeError('deployment timeout error')

    @classmethod
    def _build_model(cls, file_path: str, target_column: str, model_metric: ModelMetric, task_type: Optional[TaskType],
                     access_token: str, **kwargs):

        dai = cls._get_instance(access_token)

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

        experiment = dai.experiments.create(
            train_dataset=dataset,
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

        dai = cls._get_instance()  # This instance should already by present.
        prj = dai.projects.create(name=cls._make_project_id())
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

        cls._wait_for_deployment(mlops_client, deployment_id)

        statuses = mlops_client.deployer.deployment_status.list_deployment_statuses(
            mlops.DeployListDeploymentStatusesRequest(project_id=project_id))
        return statuses.deployment_status[0].scorer.score.url

    @classmethod
    def build(cls, file_path: str, target_column: str, model_metric: ModelMetric, task_type: Optional[TaskType],
              access_token: str, refresh_token: str, **kwargs) -> Model:
        """Builds DAI based model."""

        if refresh_token:
            access_token, refresh_token = cls._refresh_token(refresh_token, _config.oidc_provider_url,
                                                             _config.oidc_client_id, _config.oidc_client_secret)

        experiment = cls._build_model(file_path=file_path, target_column=target_column, model_metric=model_metric,
                                      task_type=task_type, access_token=access_token, **kwargs)

        if refresh_token:
            access_token, refresh_token = cls._refresh_token(refresh_token, _config.oidc_provider_url,
                                                             _config.oidc_client_id, _config.oidc_client_secret)
        elif not access_token and not refresh_token:
            raise ValueError('not token credentials for MLOps specified')

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
            access_token, _ = cls._refresh_token(refresh_token, _config.oidc_provider_url,
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

    @staticmethod
    def _encode_from_data(data: List[List]) -> Dict:
        if len(data) < 2:
            raise ValueError('invalid input format')

        return {
            'fields': data[0],
            'rows':  [[str(item) for item in row] for row in data[1:]],
        }

    @staticmethod
    def _encode_from_csv(csvfile: IO) -> Dict:
        reader = csv.reader(csvfile)
        return {
            'fields': next(reader),
            'rows': [row for row in reader],
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
    def _decode_from_deployment(cls, data) -> List[Tuple]:
        names = [cls._extract_class(name) for name in data['fields']]

        ret = []
        for row in data['score']:
            values = [float(item) for item in row]
            index = values.index(max(values))
            ret.append(tuple([names[index], *values]))

        return ret

    def predict(self, data: Optional[List[List]] = None, file_path: str = '', **kwargs) -> List[Tuple]:

        if data is not None:
            payload = self._encode_from_data(data)
        elif file_path:
            with open(file_path) as csvfile:
                payload = self._encode_from_csv(csvfile)
        else:
            raise ValueError('no data input')

        r = requests.post(self._endpoint_url, json=payload)
        r.raise_for_status()
        return self._decode_from_deployment(r.json())

    @property
    def endpoint_url(self) -> Optional[str]:
        return self._endpoint_url


def build_model(file_path: str, *, target_column: str, model_metric: ModelMetric = ModelMetric.AUTO,
                task_type: Optional[TaskType] = None, model_type: Optional[ModelType] = None,
                access_token: str = '', refresh_token: str = '', **kwargs) -> Model:
    """Trains a model.

    If `model_type` is not specified, it is inferred from the current environment. Defaults to an H2O-3 model.

    Args:
        file_path: The path to the training dataset.
        target_column: The name of the target column (the column to be predicted).
        model_metric: Optional evaluation metric to be used during modeling.
        task_type: Optional task type. Will be automatically determined if it's not specified.
        model_type: Optional model type.
        access_token: Optional access token if engine needs to be authenticated.
        refresh_token: Optional refresh token if model needs to be authenticated.
        kwargs: Optional parameters to be passed to the model builder.

    Returns:
        The Wave model.

    """

    if model_type is not None:
        if model_type == ModelType.H2O3:
            return _H2O3Model.build(file_path, target_column, model_metric, task_type, **kwargs)
        elif model_type == ModelType.DAI:
            return _DAIModel.build(file_path, target_column, model_metric, task_type, access_token, refresh_token,
                                   **kwargs)

    if _config.dai_address or _config.steam_address:
        return _DAIModel.build(file_path, target_column, model_metric, task_type, access_token, refresh_token, **kwargs)

    return _H2O3Model.build(file_path, target_column, model_metric, task_type, **kwargs)


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

    _H2O3Model._ensure()

    model = h2o.upload_model(path=file_path)
    return _H2O3Model(model)
