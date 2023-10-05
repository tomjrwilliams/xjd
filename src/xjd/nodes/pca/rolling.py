
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

import distrax
import jaxopt
import optax

import xtuples as xt

from ... import utils
from ... import xjd

from . import vanilla

# ---------------------------------------------------------------

small = 10 ** -4

@xt.nTuple.decorate(init=xjd.init_null)
class PCA_Rolling(typing.NamedTuple):
    
    n: int
    data: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[PCA_Rolling, tuple, xjd.SiteValue]: ...

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        res = data.map(
            lambda _data: vanilla.PCA.f(
                jax.numpy.transpose(_data), self.n
            )
        )
        return (
            res.map(lambda v: v[0]), # eigvals
            res.map(lambda v: v[1]), # eigvecs
        )

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xjd.init_null)
class PCA_Rolling_Encoder_Trimmed(typing.NamedTuple):
    
    n: int
    data: xjd.Location
    weights: xjd.OptionalLocation = None
    loadings: xjd.OptionalLocation = None

    clamp: float = 1.

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[PCA_Rolling_Encoder, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert self.weights is not None
        assert self.loadings is not None
        weights = self.weights.access(state)
        data = self.data.access(state)
        loadings = xt.iTuple(self.loadings.access(state)).map(
            lambda v: jax.nn.softmax(v) * self.clamp
            # unit scale, and then multiply back up to clamp
            # assume clamp < n_factors
            # so we essentially turn some off
        )
        # w = features, factors
        # l = factors
        weights = loadings.zip(weights).mapstar(
            lambda l, w: jax.numpy.multiply(
                xjd.expand_dims(l, 0, 1), w
            )
        )
        return data.map(
            jax.numpy.matmul,
            weights,
            #
        )

@xt.nTuple.decorate(init=xjd.init_null)
class PCA_Rolling_Encoder(typing.NamedTuple):
    
    n: int
    data: xjd.Location
    weights: xjd.OptionalLocation = None

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[PCA_Rolling_Encoder, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert self.weights is not None
        weights = self.weights.access(state)
        data = self.data.access(state)
        return data.map(
            jax.numpy.matmul,
            weights,
            #
        )



@xt.nTuple.decorate(init=xjd.init_null)
class PCA_Rolling_Decoder(typing.NamedTuple):
    
    factors: xjd.Location
    weights: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[PCA_Rolling_Decoder, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        weights = xt.ituple(self.weights.access(state))
        factors = self.factors.access(state).map(lambda nd: nd.T)
        return weights.map(jax.numpy.matmul, factors).map(
            lambda nd: nd.T
        )


# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xjd.init_null)
class PPCA_Rolling_NegLikelihood(typing.NamedTuple):
    
    sigma: xjd.Location
    weights: xjd.Location
    cov: xjd.Location

    # ---

    # NOTE: direct minimisation with gradient descent
    # doesn't seem to recover pca weights

    random: float = 0

    # todo put the calc into a class method
    # so can be re-used in rolling

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[PPCA_Rolling_NegLikelihood, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        sigma = self.sigma.access(state)
        weights = self.weights.access(state)
        cov = self.cov.access(state)
        res = xt.ituple(cov).map(
            vanilla.PPCA_NegLikelihood.f_apply,
            weights,
            sigma,
        )
        return jax.numpy.vstack(res.pipe(list)).sum()
        

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xjd.init_null)
class PPCA_Rolling_EM(typing.NamedTuple):
    
    sigma: xjd.Location
    weights: xjd.Location

    cov: xjd.Location

    # ---

    random: float = 0

    # NOTE: we just need covariance here

    # so we can pass this from a kernel function
    # if we want

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[PPCA_Rolling_EM, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        # https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf

        sigma = self.sigma.access(state)
        sigma_sq = jax.numpy.square(sigma)

        weights = self.weights.access(state)
        cov = self.cov.access(state) # of obs

        # use noisy_sgd instead of random
        # if self.random:
        #     key = xjd.get_location(
        #         self.loc.random(), state
        #     )
        #     weights = weights + ((
        #         jax.random.normal(key, shape=weights.shape)
        #     ) * self.random)

        # feature * feature
        S = cov
        d = weights.shape[0] # n_features

        W = weights

        mm = jax.numpy.matmul

        noise = jax.numpy.eye(weights.shape[1]) * sigma_sq
        M = mm(W.T, W) + noise

        invM = jax.numpy.linalg.inv(M)

        # M = factor by factor
        # C = feature by feature

        SW = mm(S, W)

        W_new = mm(SW, jax.numpy.linalg.inv(
            noise + mm(mm(invM, W.T), SW)
        ))
        
        sigma_sq_new = (1 / d) * jax.numpy.trace(
            jax.numpy.subtract(S, mm(
                SW, mm(invM, W_new.T)
            ))
        )

        return W_new, jax.numpy.sqrt(sigma_sq_new + small)

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xjd.init_null)
class PPCA_Rolling_Marginal_Observations(typing.NamedTuple):
    
    sigma: xjd.Location
    weights: xjd.Location
    encoder: xjd.Location
    data: xjd.Location

    cov: xjd.Location

    # ---

    random: float = 0

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[PPCA_Rolling_Marginal_Observations, tuple, xjd.SiteValue]: ...
    

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        # https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf

        sigma = self.sigma.access(state)
        sigma_sq = jax.numpy.square(sigma)

        weights = self.weights.access(state)
        cov = self.cov.access(state) # of obs

        data = self.data.access(state)
    
        # N = xjd.get_location(self.site_encoder, state).shape[0]

        # feature * feature
        S = cov
        # d = weights.shape[0] # n_features

        noise = sigma_sq * jax.numpy.eye(weights.shape[0])

        C = jax.numpy.matmul(weights, weights.T) + noise
        mu = jax.numpy.zeros(weights.shape[0])

        dist = distrax.MultivariateNormalFullCovariance(mu, C)

        return dist.log_prob(data)


@xt.nTuple.decorate(init=xjd.init_null)
class PPCA_Rolling_Conditional_Latents(typing.NamedTuple):
    
    sigma: xjd.Location
    weights: xjd.Location
    encoder: xjd.Location
    data: xjd.Location

    cov: xjd.Location

    # ---
   
    random: float = 0

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[PPCA_Rolling_Conditional_Latents, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        # https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf

        sigma = self.sigma.access(state)
        sigma_sq = jax.numpy.square(sigma)

        weights = self.weights.access(state)
        cov = self.cov.access(state) # of obs

        data = self.data.access(state)
    
        # N = xjd.get_location(self.site_encoder, state).shape[0]

        # feature * feature
        S = cov
        # d = weights.shape[0] # n_features
        factors = self.encoder.access(state)
        # replace with factors
        
        mu = jax.numpy.zeros(weights.shape[0]) # obs mu

        noise = jax.numpy.eye(weights.shape[1]) * sigma_sq

        mm = jax.numpy.matmul

        W = weights
        M = mm(weights.T, weights) + noise
        M_inv = jax.numpy.linalg.inv(M)

        # will need to expand mu to match shape of data
        dist = distrax.MultivariateNormalFullCovariance(
            mm(mm(M_inv, W.T), data - mu),
            sigma_sq * M_inv
        )
        return dist.log_prob(factors)

# ---------------------------------------------------------------
