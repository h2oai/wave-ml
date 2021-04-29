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
import uuid


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
