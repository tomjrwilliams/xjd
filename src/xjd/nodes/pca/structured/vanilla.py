


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

from .... import utils
from .... import xfactors as xf

# ---------------------------------------------------------------

# weight mask to apply to loadings directly
# eg. where we have a flat dataframe of two curves
# btu want factors that are only one or the other


@xt.nTuple.decorate(init=xf.init_null)
class Structured_PCA_Mask(typing.NamedTuple):
    
    n: int
    

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Structured_PCA_Mask, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        return ()

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class Structured_PCA_Convex(typing.NamedTuple):
    
    n: int
    

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Structured_PCA_Convex, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        return ()



@xt.nTuple.decorate(init=xf.init_null)
class Structured_PCA_Concave(typing.NamedTuple):
    
    n: int
    

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Structured_PCA_Concave, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        return ()


# ---------------------------------------------------------------

# overall factor sign
# has the same effect as the below (but if we don't care oither than the same can use below)

@xt.nTuple.decorate(init=xf.init_null)
class Structured_PCA_Sign(typing.NamedTuple):
    
    n: int
    

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Structured_PCA_Sign, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        return ()


# eg. just for factor alignment

@xt.nTuple.decorate(init=xf.init_null)
class Structured_PCA_TiedSign(typing.NamedTuple):
    
    n: int
    

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Structured_PCA_TiedSign, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        return ()

# ---------------------------------------------------------------
