
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
class Weights_Constant(typing.NamedTuple):
    """
    """

    v: float
    shape: tuple

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Weights_Constant, tuple, xjd.SiteValue]: ...
    
    def init_params(self, model, params):
        return self, jax.numpy.ones(self.shape) * self.v


@xt.nTuple.decorate(init=xjd.init_null)
class Weights_Normal(typing.NamedTuple):
    """
    """

    shape: tuple

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Weights_Normal, tuple, xjd.SiteValue]: ...
    
    def init_params(self, model, params):
        return self, utils.rand.gaussian(self.shape)


@xt.nTuple.decorate(init=xjd.init_null)
class Weights_Orthogonal(typing.NamedTuple):
    """
    """

    shape: tuple

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Weights_Orthogonal, tuple, xjd.SiteValue]: ...
    
    def init_params(self, model, params):
        return self, utils.rand.orthogonal(self.shape)

# ---------------------------------------------------------------

@xt.nTuple.decorate()
class Latent(typing.NamedTuple):
    """
    axis: None = scalar, 0 = time series, 1 = ticker
    """

    n: int
    axis: int
    data: xjd.Location
    # TODO init: collections.abc.Iterable = None

    # kwargs for specifying the init - orthogonal, gaussian, etc.

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Latent, tuple, xjd.SiteValue]:
        axis = self.axis
        obj = self.data.site().access(model)
        shape_latent = (
            (self.n,)
            if axis is None
            else (obj.shape[axis], self.n,)
        )
        assert shape_latent is not None, self
        latent = utils.rand.gaussian(shape_latent)
        return self, shape_latent, latent

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        v = site.loc.param().access(state)
        if site.masked:
            return jax.lax.stop_gradient(v)
        return v


# ---------------------------------------------------------------
