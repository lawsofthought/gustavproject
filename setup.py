from setuptools import setup

import unittest

def test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('gustav', pattern='test*.py')
    return test_suite

setup(name = "gustav",
      license = 'GNU Public License 3.0',
      version = "0.0.0",
      description = "A probabilistic topic model toolbox",
      author = "Mark Andrews",
      author_email = "mjandrews.net@gmail.com",
      packages=["gustav"],
      test_suite='setup.test_suite',
      scripts = [
        'scripts/gustave',
        'scripts/mpiexechdptm.py'
      ],
      install_requires=[
          'configobj>=5.0.6',
          'docopt>=0.6.2',
          'numpy>=1.11.2',
          'mpi4py>=2.0.0',
          'sympy>=1.0'
      ]
      )
