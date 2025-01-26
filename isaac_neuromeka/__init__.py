# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *
import os


# NRMK_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# """Absolute path to the neuromeka extension repository."""

# NRMK_ENVS_DIR = os.path.join(NRMK_ROOT_DIR, "tasks")
