
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

# below is eg. for latent gp model
# so we first group by (say) sector
# then for each, vmap gp
# then flatten back down & apply constraints (mse, ...)


# ---------------------------------------------------------------



@xt.nTuple.decorate(init=xf.init_null)
class Stack(typing.NamedTuple):
    
    locs: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Stack, tuple, xf.SiteValue]: ...    

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# flatten such a tuple from the above back down



@xt.nTuple.decorate(init=xf.init_null)
class UnStack(typing.NamedTuple):
    
    loc: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[UnStack, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# flatten such a tuple from the above back down

# ---------------------------------------------------------------



@xt.nTuple.decorate(init=xf.init_null)
class Flatten(typing.NamedTuple):
    
    locs: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Flatten, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self


@xt.nTuple.decorate(init=xf.init_null)
class UnFlatten(typing.NamedTuple):
    
    loc: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[UnFlatten, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# flatten such a tuple from the above back down
# versus the above which actually concats the arrays

# ---------------------------------------------------------------



@xt.nTuple.decorate(init=xf.init_null)
class Concatenate(typing.NamedTuple):
    
    axis: int
    loc: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Concatenate, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.loc.access(state)
        return jax.numpy.concatenate(
            data.pipe(list), axis=self.axis
            #
        )


# given shape definitions, can slice back out into tuple

@xt.nTuple.decorate(init=xf.init_null)
class UnConcatenate(typing.NamedTuple):
    
    loc: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[UnConcatenate, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# flatten such a tuple from the above back down

# ---------------------------------------------------------------
