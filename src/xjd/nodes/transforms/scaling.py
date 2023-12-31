
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
from .. import params


# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xjd.init_null)
class Expit(typing.NamedTuple):

    data: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Expit, tuple, xjd.SiteValue]: ...

    @classmethod
    def f(cls, data):
        return utils.funcs.expit(data)

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(self.data.access(state))

@xt.nTuple.decorate(init=xjd.init_null)
class Exp(typing.NamedTuple):

    data: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Exp, tuple, xjd.SiteValue]: ...

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return jax.numpy.exp(self.data.access(state))

@xt.nTuple.decorate(init=xjd.init_null)
class UnitNorm(typing.NamedTuple):

    data: xjd.Loc

    axis: int = -1

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[UnitNorm, tuple, xjd.SiteValue]: ...

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        data = self.data.access(state)
        norm = jax.numpy.sqrt(
            jax.numpy.square(data).sum(axis=self.axis)
        )
        return jax.numpy.divide(
            data, 
            xjd.expand_dims(
                norm, self.axis, data.shape[self.axis]
            )
        )

@xt.nTuple.decorate(init=xjd.init_null)
class Softmax(typing.NamedTuple):

    data: xjd.Loc

    axis: int = 0

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Softmax, tuple, xjd.SiteValue]: ...

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return jax.nn.softmax(
            self.data.access(state), axis=self.axis
        )


@xt.nTuple.decorate(init=xjd.init_null)
class Sq(typing.NamedTuple):

    data: xjd.Loc

    vmin: typing.Optional[float] = None
    vmax: typing.Optional[float] = None

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Sq, tuple, xjd.SiteValue]: ...

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        res = jax.numpy.square(self.data.access(state))
        if self.vmin is not None:
            res = self.vmin + res
        if self.vmax is not None:
            res = jax.numpy.clip(res, a_max=self.vmax)
        return res

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xjd.init_null)
class Linear1D(typing.NamedTuple):

    a: xjd.Loc
    b: xjd.Loc
    data: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Linear1D, tuple, xjd.SiteValue]: ...

    @classmethod
    def add_to_model(
        cls,
        model,
        data: xjd.OptionalLoc = None,
        a: xjd.OptionalLoc = None,
        b: xjd.OptionalLoc = None,
    ):
        assert data is not None
        if a is None:
            model, a = model.add_node(params.random.Gaussian((1,)))
        if b is None:
            model, b = model.add_node(params.random.Gaussian((1,)))
        obj = cls(a=a, b=b, data=data)
        return model.add_node(obj)

    @classmethod
    def f(cls, data, a, b):
        return utils.funcs.linear(data, a, b)

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(
            data=self.data.access(state),
            a=self.a.access(state),
            b=self.b.access(state),
        )

@xt.nTuple.decorate(init=xjd.init_null)
class Logistic(typing.NamedTuple):

    data: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Logistic, tuple, xjd.SiteValue]: ...

    @classmethod
    def f(cls, data):
        return utils.funcs.logistic(data)

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(self.data.access(state))

@xt.nTuple.decorate(init=xjd.init_null)
class Sigmoid(typing.NamedTuple):

    data: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Sigmoid, tuple, xjd.SiteValue]: ...

    @classmethod
    def f(cls, data):
        return utils.funcs.sigmoid(data)

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(self.data.access(state))

@xt.nTuple.decorate(init=xjd.init_null)
class CosineKernel(typing.NamedTuple):

    data: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Sigmoid, tuple, xjd.SiteValue]: ...

    @classmethod
    def f(cls, data):
        return utils.funcs.kernel_cosine(data)

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(self.data.access(state))


@xt.nTuple.decorate(init=xjd.init_null)
class RBFKernel(typing.NamedTuple):

    data: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Sigmoid, tuple, xjd.SiteValue]: ...

    @classmethod
    def f(cls, data):
        return utils.funcs.kernel_rbf(data)

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(self.data.access(state))

@xt.nTuple.decorate(init=xjd.init_null)
class GaussianKernel(typing.NamedTuple):

    data: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Sigmoid, tuple, xjd.SiteValue]: ...

    @classmethod
    def f(cls, data):
        return utils.funcs.kernel_gaussian(data)

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(self.data.access(state))

# ---------------------------------------------------------------
