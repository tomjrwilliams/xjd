# SPDX-FileCopyrightText: 2023-present Tom Williams <tomjrw@gmail.com>
#
# SPDX-License-Identifier: MIT

import sys
sys.path.append("./__local__")

import PATHS

if PATHS.XTUPLES not in sys.path:
    sys.path.append(PATHS.XTUPLES)

sys.path.append("./src")