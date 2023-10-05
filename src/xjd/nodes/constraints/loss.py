
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
class Maximise(typing.NamedTuple):
    
    data: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Maximise, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return -1 * data.mean()


@xt.nTuple.decorate(init=xjd.init_null)
class Minimise(typing.NamedTuple):
    
    data: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Minimise, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return data.mean()


@xt.nTuple.decorate(init=xjd.init_null)
class MinimiseSquare(typing.NamedTuple):
    
    data: xjd.Location

    dropout: float = 0

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[MinimiseSquare, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        if self.dropout:
            assert site.loc is not None
            key = site.loc.random().access(
                state, into=jax.numpy.ndarray
            )
            n = int(len(data) * (1 - self.dropout))
            data = jax.random.shuffle(key, data)[:n]
        return jax.numpy.square(data).mean()

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xjd.init_null)
class L0(typing.NamedTuple):
    
    data: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[L0, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self


@xt.nTuple.decorate(init=xjd.init_null)
class L1(typing.NamedTuple):
    
    data: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[L1, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return jax.numpy.abs(data).mean()

@xt.nTuple.decorate(init=xjd.init_null)
class L1Diag(typing.NamedTuple):
    
    data: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[L1, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return jax.numpy.abs(
            jax.numpy.diag(data)
        ).mean()

@xt.nTuple.decorate(init=xjd.init_null)
class VL1(typing.NamedTuple):
    
    data: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[L1, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = xt.ituple(self.data.access(state))
        return jax.numpy.vstack(data.map(
            lambda v: jax.numpy.abs(v).mean()
        ).pipe(list)).mean()


@xt.nTuple.decorate(init=xjd.init_null)
class L2(typing.NamedTuple):
    
    data: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[L2, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return jax.numpy.square(data).mean()

@xt.nTuple.decorate(init=xjd.init_null)
class VL2(typing.NamedTuple):
    
    data: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[L2, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = xt.ituple(self.data.access(state))
        return jax.numpy.vstack(data.map(
            lambda v: jax.numpy.square(v).mean()
        ).pipe(list)).mean()


@xt.nTuple.decorate(init=xjd.init_null)
class ElasticNet(typing.NamedTuple):
    
    data: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[ElasticNet, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xjd.init_null)
class MAbsE(typing.NamedTuple):
    
    l: xjd.Location
    r: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[MSE, tuple, xjd.SiteValue]: ...

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        l = self.l.access(state)
        r = self.r.access(state)
        return utils.funcs.loss_mabse(l, r)

@xt.nTuple.decorate(init=xjd.init_null)
class VMAbsE(typing.NamedTuple):
    
    l: xjd.Location
    r: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[MSE, tuple, xjd.SiteValue]: ...

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        l = self.l.access(state)
        r = self.r.access(state)
        return jax.numpy.vstack(
            xt.ituple(l).map(utils.funcs.loss_mabse, r).pipe(list)
        ).mean()
# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xjd.init_null)
class MSE(typing.NamedTuple):
    
    l: xjd.Location
    r: xjd.Location

    condition: xjd.OptionalLoc = None

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[MSE, tuple, xjd.SiteValue]: ...

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        l = self.l.access(state)
        r = self.r.access(state)
        if self.condition is not None:
            mask = self.condition.access(state)
            return utils.funcs.loss_mse(l, r, mask=mask)
        return utils.funcs.loss_mse(l, r)

@xt.nTuple.decorate(init=xjd.init_null)
class VMSE(typing.NamedTuple):
    
    l: xjd.Location
    r: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[MSE, tuple, xjd.SiteValue]: ...

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        l = self.l.access(state)
        r = self.r.access(state)
        return jax.numpy.vstack(
            xt.ituple(l).map(utils.funcs.loss_mse, r).pipe(list)
        ).mean()

# ---------------------------------------------------------------
