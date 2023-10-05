
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

from ... import xfactors as xf

# ---------------------------------------------------------------

# NOTE: works differently to the below?
# as we just multiply through by the mask?
@xt.nTuple.decorate(init=xf.init_null)
class Zero(typing.NamedTuple):

    data: xf.Loc
    v: numpy.ndarray

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Zero, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        return jax.numpy.multiply(
            self.data.access(state),
            self.v
        )


@xt.nTuple.decorate(init=xf.init_null)
class Positive(typing.NamedTuple):

    data: xf.Loc
    condition: typing.Union[xf.Loc, numpy.ndarray]

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Positive, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        data_pos = jax.numpy.abs(data)
        if isinstance(self.condition, xf.Loc):
            mask = self.condition.access(state)
        else:
            mask = self.condition
        return jax.numpy.where(mask, data_pos, data)
        
@xt.nTuple.decorate(init=xf.init_null)
class Negative(typing.NamedTuple):

    data: xf.Loc
    condition: typing.Union[xf.Loc, numpy.ndarray]

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Negative, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        data_neg = -1 * jax.numpy.abs(data)
        if isinstance(self.condition, xf.Loc):
            mask = self.condition.access(state)
        else:
            mask = self.condition
        return jax.numpy.where(mask, data_neg, data)

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xf.init_null)
class Where(typing.NamedTuple):

    condition: typing.Union[xf.Loc, numpy.ndarray]
    x: typing.Union[xf.Loc, numpy.ndarray]
    y: typing.Union[xf.Loc, numpy.ndarray]

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Where, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        if isinstance(self.x, xf.Loc):
            x = self.x.access(state)
        else:
            x = self.x
        if isinstance(self.y, xf.Loc):
            y = self.y.access(state)
        else:
            y = self.y
        if isinstance(self.condition, xf.Loc):
            mask = self.condition.access(state)
        else:
            mask = self.condition
        return jax.numpy.where(mask, x, y)
        
# ---------------------------------------------------------------
