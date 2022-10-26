import re
from os.path import dirname, join, realpath
from setuptools import find_packages, setup

DESCRIPTION = "Bayesian causal impact in Python"
AUTHOR = "TBD"
AUTHOR_EMAIL = "TBD"
URL = "TBD"

PROJECT_ROOT = dirname(realpath(__file__))

REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()


setup(
    name="CausalPy",
    # version=get_version(),
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    license="LICENSE",
    url=URL,
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=install_reqs,
    # tests_require=test_reqs,
)
