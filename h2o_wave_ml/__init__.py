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

"""
Python package `h2o_wave_ml` provides the API functions for automatic machine learning tasks.
"""

from .ml import build_model, get_model, save_model, load_model
from .types import Model, ModelType, ModelMetric, TaskType

__pdoc__ = {
    'config': False,
    'dai': False,
    'h2o3': False,
}
