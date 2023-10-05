
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
from ... import utils

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xf.init_null)
class Maximise(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Maximise, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return -1 * data.mean()


@xt.nTuple.decorate(init=xf.init_null)
class Minimise(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Minimise, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return data.mean()


@xt.nTuple.decorate(init=xf.init_null)
class MinimiseSquare(typing.NamedTuple):
    
    data: xf.Location

    dropout: float = 0

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[MinimiseSquare, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
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

@xt.nTuple.decorate(init=xf.init_null)
class L0(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[L0, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self


@xt.nTuple.decorate(init=xf.init_null)
class L1(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[L1, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return jax.numpy.abs(data).mean()

@xt.nTuple.decorate(init=xf.init_null)
class L1Diag(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[L1, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return jax.numpy.abs(
            jax.numpy.diag(data)
        ).mean()

@xt.nTuple.decorate(init=xf.init_null)
class VL1(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[L1, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = xt.ituple(self.data.access(state))
        return jax.numpy.vstack(data.map(
            lambda v: jax.numpy.abs(v).mean()
        ).pipe(list)).mean()


@xt.nTuple.decorate(init=xf.init_null)
class L2(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[L2, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return jax.numpy.square(data).mean()

@xt.nTuple.decorate(init=xf.init_null)
class VL2(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[L2, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = xt.ituple(self.data.access(state))
        return jax.numpy.vstack(data.map(
            lambda v: jax.numpy.square(v).mean()
        ).pipe(list)).mean()


@xt.nTuple.decorate(init=xf.init_null)
class ElasticNet(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[ElasticNet, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xf.init_null)
class MAbsE(typing.NamedTuple):
    
    l: xf.Location
    r: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[MSE, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        l = self.l.access(state)
        r = self.r.access(state)
        return utils.funcs.loss_mabse(l, r)

@xt.nTuple.decorate(init=xf.init_null)
class VMAbsE(typing.NamedTuple):
    
    l: xf.Location
    r: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[MSE, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        l = self.l.access(state)
        r = self.r.access(state)
        return jax.numpy.vstack(
            xt.ituple(l).map(utils.funcs.loss_mabse, r).pipe(list)
        ).mean()
# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xf.init_null)
class MSE(typing.NamedTuple):
    
    l: xf.Location
    r: xf.Location

    condition: xf.OptionalLoc = None

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[MSE, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        l = self.l.access(state)
        r = self.r.access(state)
        if self.condition is not None:
            mask = self.condition.access(state)
            return utils.funcs.loss_mse(l, r, mask=mask)
        return utils.funcs.loss_mse(l, r)

@xt.nTuple.decorate(init=xf.init_null)
class VMSE(typing.NamedTuple):
    
    l: xf.Location
    r: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[MSE, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        l = self.l.access(state)
        r = self.r.access(state)
        return jax.numpy.vstack(
            xt.ituple(l).map(utils.funcs.loss_mse, r).pipe(list)
        ).mean()

# ---------------------------------------------------------------
