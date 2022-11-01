from os.path import dirname, join, realpath

from setuptools import find_packages, setup

DESCRIPTION = "Causal inference for quasi-experiments in Python"
AUTHOR = "Benjamin T. Vincent"
URL = "https://github.com/pymc-labs/CausalPy"

PROJECT_ROOT = dirname(realpath(__file__))

REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

setup(
    name="CausalPy",
    version="0.0.2",
    maintainer=AUTHOR,
    description=DESCRIPTION,
    license="LICENSE",
    url=URL,
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=install_reqs,
    # tests_require=test_reqs,
)
