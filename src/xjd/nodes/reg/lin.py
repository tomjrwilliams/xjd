
from __future__ import annotations

import operator
import collections
# import collections.abc
import functools
import itertools

import typing
import datetime

import numpy
import pandas

import jax
import jax.numpy
import jax.numpy.linalg

import jaxopt
import optax

import xtuples as xt

from ... import xjd
from .. import pca

# ---------------------------------------------------------------

Lin_Reg = pca.vanilla.PCA_Encoder

# lasso / ridge / elastic net
# via coordinate descent type algos (?)

# can also implement by combination of mse and relevant norm loss on weights

# ---------------------------------------------------------------
