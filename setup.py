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

import setuptools

with open('DESCRIPTION.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='h2o_wave_ml',
    version='0.2.1',
    author='H2O.ai',
    author_email='support@h2o.ai',
    description='AutoML for Wave Apps',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/h2oai/wave-ml',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Operating System :: POSIX :: Linux',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
    ],
    python_requires='>=3.6.1',
    install_requires=[
        'datatable==0.11.1',
        'h2o==3.32.0.2',
    ]
)
