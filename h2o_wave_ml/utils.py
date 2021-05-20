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

import sys
from typing import Tuple, Dict, List
import uuid
from urllib.parse import urljoin

try:
    import h2osteam
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
