import os
import sys

from setuptools import find_packages, setup

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
README_FILE = os.path.join(PROJECT_ROOT, "README.md")
VERSION_FILE = os.path.join(PROJECT_ROOT, "bambi", "version.py")
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")


def get_long_description():
    with open(README_FILE, encoding="utf-8") as f:
        return f.read()


# get version
sys.path.insert(0, os.path.abspath("../"))
from causalpy.version import __version__

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

setup(
    name="CausalPy",
    version=__version__,
    description="Causal inference for quasi-experiments in Python",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/pymc-labs/CausalPy",
    packages=find_packages(exclude=["tests", "test_*"]),
    package_data={"causalpy": ["data/*.csv"]},
    python_requires=">=3.8",
    maintainer="Benjamin T. Vincent",
    install_requires=install_reqs,
    # tests_require=test_reqs,
)
