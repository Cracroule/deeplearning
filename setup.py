__author__ = 'rpil'

from setuptools import setup, find_packages
from distutils.core import setup

setup(
    version='0.0.0.1',
    name='first_ml_attempt',
    description='personal project about machine learning',
    license='rpil',
    package_dir={},
    packages=['ml'],
    scripts=['scripts/example.py'],
    install_requires=[]
)
