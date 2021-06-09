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
import sys
from typing import Tuple, Dict, List
import uuid
from urllib.parse import urljoin

try:
    import h2osteam
    import mlops
except ModuleNotFoundError:
    pass
import requests

from .config import _config


def _make_id() -> str:
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


def _connect_to_steam(access_token: str = ''):

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


def list_dai_instances(access_token: str = '', refresh_token: str = '') -> List[Dict]:
    """Gets a list of all available Driverless instances.

    A token is required to authenticate with Steam if `H2O_WAVE_ML_STEAM_REFRESH_TOKEN` is not set.

    Args:
        access_token: Optional access token to authenticate with Steam.
        refresh_token: Optional refresh token to authenticate with Steam.

    Returns:
        A list of Driverless instances. The list contains a dictionary with `name`, `status` and `created_by` items.

    """

    if refresh_token:
        access_token, refresh_token = _refresh_token(refresh_token, _config.oidc_provider_url,
                                                     _config.oidc_client_id, _config.oidc_client_secret)
    _connect_to_steam(access_token)
    instances = h2osteam.api().get_driverless_instances()
    return [{'id': i['id'], 'name': i['name'], 'status': i['status'],
             'created_by': i['created_by'], 'version': i['version']} for i in instances]


def list_dai_multinodes(access_token: str = '', refresh_token: str = '') -> List[str]:
    """Gets a list of all available Driverless multinode instances.

    A token is required to authenticate with Steam if `H2O_WAVE_ML_STEAM_REFRESH_TOKEN` is not set.

    Args:
        access_token: Optional access token to authenticate with Steam.
        refresh_token: Optional refresh token to authenticate with Steam.

    Returns:
        A list of Driverless multinode instances.

    """

    if refresh_token:
        access_token, refresh_token = _refresh_token(refresh_token, _config.oidc_provider_url,
                                                     _config.oidc_client_id, _config.oidc_client_secret)
    _connect_to_steam(access_token)
    multinodes = h2osteam.api().get_driverless_multinodes()
    return [m['name'] for m in multinodes]


def _get_autodoc_artifact(mlops_client, model_id: str):  # -> Optional[mlops.StorageArtifact]:
    response = mlops_client.storage.artifact.list_entity_artifacts(
        mlops.StorageListEntityArtifactsRequest(entity_id=model_id))
    for artifact in response.artifact:
        if artifact.type == 'dai/autoreport':
            return artifact
    return None


def save_autodoc(project_id: str, output_dir_path: str, access_token: str = '', refresh_token: str = '') -> str:
    """Saves an autodoc document from MLOps to the given location.

    Access or refresh token is required in order to connect to MLOps.

    Args:
        project_id: MLOps project id.
        output_dir_path: A directory where the doc will be saved.
        access_token: Access token to authenticate with MLOps.
        refresh_token: Refresh token to authenticate with MLOps.

    Returns:
        The file path to the saved report.

    """

    if refresh_token:
        access_token, refresh_token = _refresh_token(refresh_token, _config.oidc_provider_url,
                                                     _config.oidc_client_id, _config.oidc_client_secret)

    mlops_client = mlops.Client(gateway_url=_config.mlops_gateway, token_provider=lambda: access_token)
    response = mlops_client.storage.experiment.list_experiments(
        mlops.StorageListExperimentsRequest(project_id=project_id))

    # There should be a strategy to pick the right experiment instead of picking a zeroth one.
    # Wave ML creates a one project per experiment.
    model_id = response.experiment[0].id

    autodoc = _get_autodoc_artifact(mlops_client, model_id)
    if autodoc is None:
        raise ValueError(f'no autodoc available for model {model_id}')

    file_name = f'autodoc_{model_id}.docx'
    autodoc_path = str(Path(output_dir_path, file_name))

    with open(autodoc_path, 'wb') as f:
        mlops_client.storage.artifact.download_artifact(artifact_id=autodoc.id, file=f)

    return autodoc_path
