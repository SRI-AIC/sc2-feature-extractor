#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='sc2-feature-extractor',
      version='1.0',
      description='StarCraft II high-level feature extractor and replay visualizer',
      url='https://github.com/SRI-AIC/sc2-feature-extractor',
      author='Pedro Sequeira',
      author_email='pedro.sequeira@sri.com',
      packages=find_packages(),
      scripts=[
      ],
      install_requires=[
          'protobuf == 3.20.1',
          'pysc2',
          'numpy',
          'pandas',
          'jsonpickle',
          'scipy',
          'absl-py',
          'tqdm',
          'matplotlib',
          'plotly',
          'kaleido',
          'scikit-video >= 1.1.11',
          'joblib'
      ],
      extras_require={
          'macos': [
              'pyobjc-framework-Quartz'
          ],
          'windows': [
              'pywin32'
          ],
      },
      zip_safe=True
      )
