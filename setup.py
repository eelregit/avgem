#!/usr/bin/env python
from setuptools import setup
import avgem

setup(
    name = "avgem",
    version = "0.0.1",
    description = avgem.__doc__,
    url = "https://github.com/eelregit/avgem",
    author = "Yin Li",
    author_email = "eelregit@gmail.com",
    license = "GPLv3",
    keywords = "average binning interpolation",
    packages = ["avgem"],
    install_requires = ["numpy", "scipy"],
)
