from setuptools import find_packages, setup

setup(
    name='mypythonlib',
    packages=find_packages(
      include=['src']
    ),
    version='0.1.0',
    description='Aprendizagem Automática',
    author='Vaux Gomes',
    license='Apache Licence',
)