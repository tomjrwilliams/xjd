
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

# ---------------------------------------------------------------

# NOTE: works differently to the below?
# as we just multiply through by the mask?
@xt.nTuple.decorate(init=xjd.init_null)
class Zero(typing.NamedTuple):

    data: xjd.Loc
    v: numpy.ndarray

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Zero, tuple, xjd.SiteValue]: ...

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        return jax.numpy.multiply(
            self.data.access(state),
            self.v
        )

# ---------------------------------------------------------------

def where(mask, if_yes, if_no):
    not_mask = 1 + (mask * -1.)
    return jax.numpy.multiply(
        mask, if_yes
    ) + jax.numpy.multiply(
        not_mask, if_no
    )

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xjd.init_null)
class Positive(typing.NamedTuple):

    data: xjd.Loc
    condition: typing.Union[xjd.Loc, numpy.ndarray]

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
        data_pos = jax.numpy.abs(data)
        if isinstance(self.condition, xjd.Loc):
            mask = self.condition.access(state)
        else:
            mask = self.condition
        return where(mask, data_pos, data)
        
@xt.nTuple.decorate(init=xjd.init_null)
class Negative(typing.NamedTuple):

    data: xjd.Loc
    condition: typing.Union[xjd.Loc, numpy.ndarray]

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Negative, tuple, xjd.SiteValue]: ...

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        data_neg = -1 * jax.numpy.abs(data)
        if isinstance(self.condition, xjd.Loc):
            mask = self.condition.access(state)
        else:
            mask = self.condition
        return where(mask, data_neg, data)

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xjd.init_null)
class Where(typing.NamedTuple):

    condition: typing.Union[xjd.Loc, numpy.ndarray]
    x: typing.Union[xjd.Loc, numpy.ndarray]
    y: typing.Union[xjd.Loc, numpy.ndarray]

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Where, tuple, xjd.SiteValue]: ...

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        if isinstance(self.x, xjd.Loc):
            x = self.x.access(state)
        else:
            x = self.x
        if isinstance(self.y, xjd.Loc):
            y = self.y.access(state)
        else:
            y = self.y
        if isinstance(self.condition, xjd.Loc):
            mask = self.condition.access(state)
        else:
            mask = self.condition
        return jax.numpy.where(mask, x, y)
        
# ---------------------------------------------------------------
