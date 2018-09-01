#!/usr/bin/env python
from setuptools import setup
import avgem

def find_version(path):
    with open(path, 'r') as fp:
        file = fp.read()
    import re
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                            file, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Version not found")

setup(
    name = "avgem",
    version = find_version("avgem/__init__.py"),
    description = avgem.__doc__,
    url = "https://github.com/eelregit/avgem",
    author = "Yin Li",
    author_email = "eelregit@gmail.com",
    license = "GPLv3",
    keywords = "average binning",
    packages = ["avgem"],
    install_requires = ["numpy", "scipy>=0.19.0"],
)
