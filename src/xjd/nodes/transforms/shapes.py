
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

# below is eg. for latent gp model
# so we first group by (say) sector
# then for each, vmap gp
# then flatten back down & apply constraints (mse, ...)


# ---------------------------------------------------------------



@xt.nTuple.decorate(init=xjd.init_null)
class Stack(typing.NamedTuple):
    
    locs: xt.iTuple

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Stack, tuple, xjd.SiteValue]: ...    

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# flatten such a tuple from the above back down



@xt.nTuple.decorate(init=xjd.init_null)
class UnStack(typing.NamedTuple):
    
    loc: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[UnStack, tuple, xjd.SiteValue]: ...

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# flatten such a tuple from the above back down

# ---------------------------------------------------------------



@xt.nTuple.decorate(init=xjd.init_null)
class Flatten(typing.NamedTuple):
    
    locs: xt.iTuple

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Flatten, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self


@xt.nTuple.decorate(init=xjd.init_null)
class UnFlatten(typing.NamedTuple):
    
    loc: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[UnFlatten, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# flatten such a tuple from the above back down
# versus the above which actually concats the arrays

# ---------------------------------------------------------------



@xt.nTuple.decorate(init=xjd.init_null)
class Concatenate(typing.NamedTuple):
    
    axis: int
    loc: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Concatenate, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.loc.access(state)
        return jax.numpy.concatenate(
            data.pipe(list), axis=self.axis
            #
        )


# given shape definitions, can slice back out into tuple

@xt.nTuple.decorate(init=xjd.init_null)
class UnConcatenate(typing.NamedTuple):
    
    loc: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[UnConcatenate, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# flatten such a tuple from the above back down

# ---------------------------------------------------------------
