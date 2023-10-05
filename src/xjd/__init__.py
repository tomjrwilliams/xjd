# SPDX-FileCopyrightText: 2023-present Tom Williams <tomjrw@gmail.com>
#
# SPDX-License-Identifier: MIT

import os
import sys
import pathlib

if pathlib.Path(os.getcwd()).parts[-1] == "xjd":
    sys.path.append("./__local__")

    import PATHS

    if PATHS.XTUPLES not in sys.path:
        sys.path.append(PATHS.XTUPLES)

from . import utils
from . import nodes

from .nodes import *
from .xjd import *
