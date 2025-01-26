# Copyright (c) 2025, Neuromeka Co., Ltd.
# All rights reserved.
#
# Apache-2.0 License

import os
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Installation operation
setup(
    name="isaac_neuromeka",
    author="NRMK AI Lab",
    maintainer="Joonho Lee, Yunho Kim",
    maintainer_email="...",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    include_package_data=True,
    python_requires=">=3.10",
    packages=["isaac_neuromeka"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.2.0",
    ],
    zip_safe=False,
)