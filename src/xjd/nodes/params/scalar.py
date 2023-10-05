
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

@xt.nTuple.decorate()
class Scalar(typing.NamedTuple):

    v: numpy.ndarray

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Scalar, tuple, xf.SiteValue]:
        # TODO
        return self, (), jax.numpy.array(self.v)

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        v = site.loc.param().access(state)
        if site.masked:
            return jax.lax.stop_gradient(v)
        return v


@xt.nTuple.decorate()
class VScalar(typing.NamedTuple):

    data: xf.Location
    v: numpy.ndarray

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[VScalar, tuple, xf.SiteValue]:
        target: xf.Site = self.data.access(model, into=xf.Site)
        assert target.shape is not None # sigh, oh mypy
        n = len(target.shape)
        return self, tuple(
            self.v.shape for _ in range(n)
        ), xt.iTuple.range(n).map(
            lambda i: jax.numpy.array(self.v)
        )

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        v = site.loc.param().access(state)
        if site.masked:
            return v.map(jax.lax.stop_gradient)
        return v

# ---------------------------------------------------------------