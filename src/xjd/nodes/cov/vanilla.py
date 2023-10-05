
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
class Cov(typing.NamedTuple):

    data: xf.Loc
    exists: xf.OptionalLoc = None
    
    shrinkage: typing.Optional[str] = None

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Cov, tuple, xf.SiteValue]:
        vs = self.data.site().access(model)
        n = vs.shape[1]
        return self, (n, n,), ()

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        if self.exists is None:
            return utils.funcs.cov(data)
        else:
            # assume data is zero mask where exists
            # data = n_points, n_features
            exists = self.exists.access(state)
            return utils.funcs.cov(data, exists=exists)

@xt.nTuple.decorate(init=xf.init_null)
class VCov(typing.NamedTuple):
    
    data: xf.Loc
    exists: xf.OptionalLoc = None
    
    shrinkage: typing.Optional[str] = None

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[VCov, tuple, xf.SiteValue]:
        vs = self.data.site().access(model)
        shape = vs.map(lambda v: v.shape[1]).map(lambda n: (n, n,))
        return self, shape, ()

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        if self.exists is None:
            return data.map(utils.funcs.cov)
        else:
            exists = self.exists.access(state)
            return data.zip(exists).mapstar(utils.funcs.cov)
        
# TODO move shrinkage here

# also add cov_with_missing where we have different number of samples not none in a given df
# so pairwise calculate with different denominator (with a minimin sampling number)


# can even then have shrinkage maybe calcualted per name?

# ---------------------------------------------------------------
