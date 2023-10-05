
from __future__ import annotations
import enum

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
from ... import utils

# ---------------------------------------------------------------

# TODO: if eg. learning factor path over specific dates
# then here is where we encode that restriction
# specific stock universe, etc.


@xt.nTuple.decorate()
class NDArray(typing.NamedTuple):

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[NDArray, tuple, xjd.SiteValue]:
        assert isinstance(data, numpy.ndarray), type(data)
        return self, data.shape, ()
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert isinstance(data, numpy.ndarray), type(data)
        return jax.numpy.array(data)

# ---------------------------------------------------------------
