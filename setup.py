# -*- coding: utf-8 -*-
# ******************************************************
# aitk.networks: Keras model wrapper with visualizations
#
# Copyright (c) 2021 Douglas S. Blank
#
# https://github.com/ArtificialInteligenceToolkit/aitk.networks
#
# ******************************************************

"""
aitk.networks setup
"""
import io
import os

import setuptools

HERE = os.path.abspath(os.path.dirname(__file__))

# The name of the project
name = "aitk.networks"


# Get our version
def get_version(file, name="__version__"):
    """Get the version of the package from the given file by
    executing it and extracting the given `name`.
    """
    path = os.path.realpath(file)
    version_ns = {}
    with io.open(path, encoding="utf8") as f:
        exec(f.read(), {}, version_ns)
    return version_ns[name]


version = get_version(os.path.join(HERE, "aitk/networks/_version.py"))

with open("README.md", "r") as fh:
    long_description = fh.read()

setup_args = dict(
    name=name,
    version=version,
    url="https://github.com/ArtificialIntelligenceToolkit/%s" % name,
    author="Douglas Blank",
    description="A Keras model wrapper with visualizations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_namespace_packages(include=['aitk.*']),
    install_requires=["setuptools", "Pillow", "numpy", "tensorflow", "matplotlib", "aitk.utils"],
    extras_require={"jupyter": ["ipywidgets", "IPython"], "image": ["cairosvg"]},
    python_requires=">=3.6",
    license="BSD-3-Clause",
    platforms="Linux, Mac OS X, Windows",
    keywords=["robot", "simulator", "jupyter", "python"],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Framework :: Jupyter",
    ],
)

if __name__ == "__main__":
    setuptools.setup(**setup_args)
