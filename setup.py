import os

from setuptools import find_packages, setup

__version__ = None


# utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="MolNet",
    version=__version__,
    author="Kevin Spiekermann",
    description="Uses graph networks prediction properties of molecules or reactions.",
    url="https://github.com/kspieks/molnet",
    packages=find_packages(),
    long_description=read('README.md'),
)
