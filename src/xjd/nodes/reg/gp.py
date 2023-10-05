
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

from .. import cov

# ---------------------------------------------------------------



# NOTE: this is just a kernel, so is currently a misnomer
@xt.nTuple.decorate()
class GP_RBF(typing.NamedTuple):

    # sigma: float
    features: xf.Location
    # data: xf.Location

    # optional weights?
    # optional mean

    n: typing.Optional[int] = None
    # shape determines how many samples
    # eg. if using a gp for three factors
    # if mean provided, assert mean.shape == n

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[GP_RBF, tuple, xf.SiteValue]:
        return self, (), (
            jax.numpy.ones(1),
            jax.numpy.ones(1),
        )

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:

        assert site.loc is not None
        l, sigma = site.loc.param().access(state)
        
        # = n_variables, n_latents
        features = self.features.access(state)

        n_variables = features.shape[0]
        n_features = features.shape[1]

        # data = self.data.access(state)

        # assert data.shape[1] == n_variables

        features_matrix = xf.expand_dims(
            features, axis = 0, size = features.shape[0]
        )

        features_l = jax.numpy.reshape(
            features_matrix,
            (n_variables ** 2, n_features,)
        )
        # left = iterates through variables
        features_r = jax.numpy.reshape(
            jax.numpy.transpose(features_matrix, (1, 0, 2,)),
            (n_variables ** 2, n_features,)
        )
        # right = blocks of same variable's latents
        kernel = utils.funcs.kernel_rbf(
            utils.funcs.diff_euclidean(
                features_r, features_l
            ),
            l,
            sigma,
        )
        res = jax.numpy.reshape(
            kernel, (n_variables, n_variables,)
        )

        # assert (cov == cov.T).all()

        return res

# ---------------------------------------------------------------

# aka white noise

@xt.nTuple.decorate()
class GP_Kernel_Sigmoid(typing.NamedTuple):

    sigma: float
    # or variance?
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[GP_Kernel_Sigmoid, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# ---------------------------------------------------------------
 

@xt.nTuple.decorate()
class GP_Kernel_SquaredExp(typing.NamedTuple):

    length_scale: float
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[GP_Kernel_SquaredExp, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self
        

@xt.nTuple.decorate()
class GP_Kernel_OU(typing.NamedTuple):

    length_scale: float
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[GP_Kernel_OU, tuple, xf.SiteValue]: ...
    
    @classmethod
    def f(cls, features_l, features_r, sigma, l):
        sigma_sq = jax.numpy.square(sigma)
        # l_2_sq = 2 * jax.numpy.square(l)
        norms = utils.funcs.diff_euclidean(features_l, features_r)
        return jax.numpy.exp(
            -1 * (jax.numpy.square(norms) / l)
        ) * sigma_sq

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self
     
# ---------------------------------------------------------------
   

@xt.nTuple.decorate()
class GP_Kernel_RationalQuadratic(typing.NamedTuple):

    length_scale: float
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[GP_Kernel_RationalQuadratic, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# ---------------------------------------------------------------
