import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='h2o_wave_ml',
    version='0.1.0',
    author='Peter Szabo',
    author_email='peter.szabo@h2o.ai',
    description='Machine Learning API for Wave',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/h2oai/wave-ml',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: POSIX :: Linux',
    ],
    python_requires='>=3.6.1',
    install_requires=[
        'h2o-wave',
        'datatable==0.11.1',
        'h2o==3.32.0.2',
    ]
)
