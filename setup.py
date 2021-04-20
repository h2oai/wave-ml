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
    version='0.4.0',
    author='H2O.ai',
    author_email='support@h2o.ai',
    description='AutoML for Wave Apps',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/h2oai/wave-ml',
    packages=setuptools.find_packages(exclude=('examples',)),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Operating System :: POSIX :: Linux',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
    ],
    python_requires='>=3.6.1',
    install_requires=[
        'datatable',
        'h2o==3.32.0.2',
        'driverlessai',
        'h2osteam@https://enterprise-steam.s3.amazonaws.com/release/1.8.2/python/h2osteam-1.8.2-py2.py3-none-any.whl',
        'mlops-client@https://s3.amazonaws.com/artifacts.h2o.ai/releases/ai/h2o/mlops/rel-0.40.1/6/mlops_client-0.40.1%2Bea66172.rel0.40.1.12-py2.py3-none-any.whl',
        'requests',
    ]
)
