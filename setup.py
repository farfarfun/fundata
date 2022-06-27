import sys
from os import path

from notebuild.tool import read_version
from setuptools import find_packages, setup
version_path = path.join(path.abspath(path.dirname(__file__)), 'script/__version__.md')
version = read_version(version_path)

install_requires = ['notetool', 'notedrive']

setup(name='notedata',
      version=version,
      description='notedata',
      author='euler',
      author_email='1007530194@qq.com',
      url='https://github.com/1007530194',

      packages=find_packages(),
      package_data={"": ["*.db"]},
      install_requires=install_requires
      )
