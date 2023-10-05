
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
from ... import xfactors as xf

# ---------------------------------------------------------------

# from jax.config import config 
# config.update("jax_debug_nans", True) 

def calc_loadings(eigvals, eigvecs):
    return jax.numpy.multiply(
        # jax.numpy.resize(
        #     jax.numpy.expand_dims(eigvals, 0),
        #     eigvecs.shape,
        # ),
        xf.expand_dims_like(eigvals, axis=0, like=eigvecs),
        eigvecs,
    )

# ---------------------------------------------------------------

@xt.nTuple.decorate()
class PCA(typing.NamedTuple):
    
    n: int
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[PCA, tuple, xf.SiteValue]:
        return self, (
            self.data.site().access(model).shape[1],
            self.n,
        ), ()

    @classmethod
    def f(cls, data, n):
        eigvals, weights = jax.numpy.linalg.eig(jax.numpy.cov(data))
        order = numpy.flip(numpy.argsort(eigvals))[:n]
        return eigvals[order], weights[..., order]

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return self.f(jax.numpy.transpose(data), self.n)

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class PCA_Encoder(typing.NamedTuple):
    
    n: int

    data: xf.Location
    weights: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[PCA_Encoder, tuple, xf.SiteValue]:
        return self, self.weights.site().access(model).shape, ()

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert self.weights is not None
        weights = self.weights.access(state)
        data = self.data.access(state)
        return jax.numpy.matmul(data, weights)



@xt.nTuple.decorate()
class PCA_Decoder(typing.NamedTuple):

    factors: xf.Location
    weights: xf.OptionalLocation = None

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[PCA_Decoder, tuple, xf.SiteValue]:
        # TODO
        return self, (), ()

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert self.weights is not None
        data = self.factors.access(state)
        weights = self.weights.access(state)
        return jax.numpy.matmul(weights, data.T).T


# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class PPCA_NegLikelihood(typing.NamedTuple):
    
    sigma: xf.Location
    weights: xf.Location
    cov: xf.Location

    # ---

    # NOTE: direct minimisation with gradient descent
    # doesn't seem to recover pca weights

    noise: float = 0.01

    # todo put the calc into a class method
    # so can be re-used in rolling

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[PPCA_NegLikelihood, tuple, xf.SiteValue]: ...

    @classmethod
    def f_apply(
        cls, S, W, sigma
    ):
        
        sigma_sq = jax.numpy.square(sigma)

        noise = jax.numpy.eye(W.shape[0]) * sigma_sq
        C = jax.numpy.add(
            jax.numpy.matmul(W, W.T),
            noise
        )
        # noise = jax.numpy.eye(weights.shape[1]) * sigma_sq
        # M = jax.numpy.add(
        #     jax.numpy.matmul(weights.T, weights),
        #     noise
        # )
        # invM = jax.numpy.linalg.inv(M)

        detC = jax.numpy.linalg.det(C)
        invC = jax.numpy.linalg.inv(C)

        invC_S = jax.numpy.matmul(invC, S)
        trace_invC_S = jax.numpy.trace(invC_S)

        # max this (dropped the * N)
        L = - (
            # + (d * jax.numpy.log(2 * numpy.pi))
            + jax.numpy.log(detC) + trace_invC_S
        ) / 2
        # so min neg(L)

        return -L

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        # https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf

        sigma = self.sigma.access(state)

        weights = self.weights.access(state)
        cov = self.cov.access(state) # of obs
    
        # N = self.site_encoder.access(state).shape[0]

        # feature * feature
        S = cov
        # d = weights.shape[0] # n_features

        if self.noise:
            assert site.loc is not None
            key = site.loc.random().access(
                state, into=jax.numpy.ndarray
            )
            weights = weights + ((
                jax.random.normal(key, shape=weights.shape)
            ) * self.noise)
    
        W = weights

        return self.f_apply(S, W, sigma)
        

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class PPCA_EM(typing.NamedTuple):
    
    sigma: xf.Location
    weights: xf.Location
    cov: xf.Location

    # ---

    random: float = 0

    # NOTE: we just need covariance here

    # so we can pass this from a kernel function
    # if we want

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[PPCA_EM, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        # https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf

        sigma = self.sigma.access(state)
        sigma_sq = jax.numpy.square(sigma)

        weights = self.weights.access(state)
        cov = self.cov.access(state) # of obs

        # use noisy_sgd instead of random
        # if self.random:
        #     key = self.loc.random().access(state)
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


@xt.nTuple.decorate(init=xf.init_null)
class PPCA_Marginal_Observations(typing.NamedTuple):
    
    sigma: xf.Location
    weights: xf.Location
    encoder: xf.Location
    data: xf.Location
    cov: xf.Location

    # ---

    random: float = 0

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[PPCA_Marginal_Observations, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        # https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf

        sigma = self.sigma.access(state)
        sigma_sq = jax.numpy.square(sigma)

        weights = self.weights.access(state)
        cov = self.cov.access(state) # of obs

        data = self.data.access(state)
    
        # N = self.site_encoder.access(state).shape[0]

        # feature * feature
        S = cov
        # d = weights.shape[0] # n_features

        noise = sigma_sq * jax.numpy.eye(weights.shape[0])

        C = jax.numpy.matmul(weights, weights.T) + noise
        mu = jax.numpy.zeros(weights.shape[0])

        dist = distrax.MultivariateNormalFullCovariance(mu, C)

        return dist.log_prob(data)

small = 10 ** -4

@xt.nTuple.decorate(init=xf.init_null)
class PPCA_Conditional_Latents(typing.NamedTuple):
    
    sigma: xf.Location
    weights: xf.Location
    encoder: xf.Location
    data: xf.Location
    cov: xf.Location

    # ---

    random: float = 0

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[PPCA_Conditional_Latents, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        # https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf

        sigma = self.sigma.access(state)
        sigma_sq = jax.numpy.square(sigma)

        weights = self.weights.access(state)
        cov = self.cov.access(state) # of obs

        data = self.data.access(state)
    
        # N = self.site_encoder.access(state).shape[0]

        # feature * feature
        S = cov
        # d = weights.shape[0] # n_features
        factors = self.encoder.access(state)
        
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
