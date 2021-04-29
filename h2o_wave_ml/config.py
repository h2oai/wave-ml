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

from typing import Any, Union
import os


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
