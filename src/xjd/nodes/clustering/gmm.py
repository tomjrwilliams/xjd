
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

import jax.scipy.special

digamma = jax.scipy.special.digamma


# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class GMM(typing.NamedTuple):
    
    n: int
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[GMM, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        # https://en.wikipedia.org/wiki/EM_algorithm_and_GMM_model
        assert False

# ---------------------------------------------------------------

small = 10 ** -4

@xt.nTuple.decorate(init=xf.init_null)
class Likelihood_Separability(typing.NamedTuple):
    
    k: int
    data: xf.Location

    mu: xf.Location
    cov: xf.Location
    # probs: xf.Location

    noise: typing.Optional[float] = 0.1

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Likelihood_Separability, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:

        # https://en.wikipedia.org/wiki/EM_algorithm_and_GMM_model

        data = self.data.access(state)
        mu = self.mu.access(state)
        cov = self.cov.access(state)
        # probs = self.probs.access(state)

        cov = jax.numpy.matmul(
            jax.numpy.transpose(cov, (0, 2, 1)),
            cov,
        )
        
        if self.noise:
            assert site.loc is not None
            key = site.loc.random().access(
                state, into=jax.numpy.ndarray
            )
            noise = ((
                jax.random.normal(key, shape=cov.shape[:-1])
            ) * self.noise)
            diag_noise = jax.numpy.multiply(
                xf.expand_dims(
                    jax.numpy.eye(cov.shape[-1]),
                    0, 
                    noise.shape[0]
                ),
                xf.expand_dims(noise, 1, 1), 
            )
            cov = cov + jax.numpy.abs(diag_noise)

        mu_ = xf.expand_dims(mu, 1, data.shape[0])
        data_ = xf.expand_dims(data, 0, mu.shape[0])
        cov_ = xf.expand_dims(
            jax.numpy.linalg.inv(cov), 1, data.shape[0]
        )

        mu_diff = xf.expand_dims(
            jax.numpy.subtract(data_, mu_),
            2, 
            1
        )
        mu_diff_T = jax.numpy.transpose(
            mu_diff,
            (0, 1, 3, 2),
        )

        det = jax.numpy.linalg.det(
            cov,
            #  axis1=1, axis2=2
        )

        norm = 1 / (
            jax.numpy.sqrt(det) * (
                (2 * numpy.pi) ** (data.shape[1] / 2)
            )
        )

        w_unnorm = jax.numpy.exp(
            -(1/2) * (
                jax.numpy.matmul(
                    jax.numpy.matmul(
                        mu_diff,
                        cov_
                    ),
                    mu_diff_T,
                )
            )
        ).squeeze().squeeze().T
        # n_data, n_clusters (?)

        w = jax.numpy.multiply(
            w_unnorm,
            xf.expand_dims(norm, 0, data.shape[0])
        )

        log_likelihood = jax.numpy.log(w.sum(axis=1)).mean()

        max_w = w.T[jax.numpy.argmax(w, axis = 1)]
        mean_w = w.sum(axis=1) - max_w

        separability = (max_w - mean_w).mean()

        data_probs = w

        return data_probs, log_likelihood, separability

# ---------------------------------------------------------------
