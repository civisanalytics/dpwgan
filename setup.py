import os
from setuptools import find_packages, setup

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

setup(
    name='dpwgan',
    version='0.1.0',
    author='Civis Analytics, Inc.',
    author_email='opensource@civisanalytics.com',
    packages=find_packages(),
    url='https://github.com/civisanalytics/dpwgan',
    description=('Differentially Private Generative Adversarial Network'
                 ' in PyTorch'),
    long_description=open(os.path.join(THIS_DIR, 'README.md')).read(),
    include_package_data=True,
    license="BSD-3",
    install_requires=['numpy>=1.17',
                      'pandas>=0.25',
                      'torch>=1.3']
)
