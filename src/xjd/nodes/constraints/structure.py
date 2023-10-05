
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
from ... import utils

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xjd.init_null)
class Positive(typing.NamedTuple):
    
    data: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Positive, tuple, xjd.SiteValue]: ...

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return jax.numpy.square(jax.numpy.clip(data, a_max = 0)).mean()

# ---------------------------------------------------------------
