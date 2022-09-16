#!/usr/bin/env python

"""The setup script."""

import os
from setuptools import setup, find_packages, dist


import numpy as np

requirements = [
    "numpy",
]



setup(
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        "setuptools>=18.0",
        "numpy",
    ],
    author="Daniel Kramer",
    author_email="drk98@nau.edu",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="A Python package for running LS on a GPU",
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    keywords="GPULS",
    name="GPULS",
    packages=find_packages(include=["gpuls", "gpuls.*"]),
    url="https://github.dev/SNAPS-Alert-Broker",
    version='1.0.0',
    zip_safe=False,
)
