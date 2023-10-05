
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

from .... import utils
from .... import xjd

# ---------------------------------------------------------------

# eg. latent features = equity sectors
# n_latents > factors, zero weights on extras (noise factors)

# so each sector (per latent_factor) has weighting
# on the equivalent index loading factor
# with 1 in the features (tickers) in that sector, zero elsewhere


@xt.nTuple.decorate(init=xjd.init_null)
class PCA_Rolling_LatentWeightedMean_MSE(typing.NamedTuple):
    
    # sites
    weights_pca: xjd.Location
    weights_structure: xjd.Location
    latents: xjd.Location

    # assume feature * factor minimum

    bias_factor: bool = True
    share_factors: bool = True

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[PCA_Rolling_LatentWeightedMean_MSE, tuple, xjd.SiteValue]: ...

    def f_apply(
        self, weights_pca, weights_structure, latents, 
    ):
        # latents = n_latents, n_latent_features
        # weights_pca = features * factors 
        # weights_structure = n_latents * n_features

        if self.share_factors:
            #  and not self.bias_factor:
            weights_structure = xjd.expand_dims(
                weights_structure, 0, weights_pca.shape[1]
            )
        # elif self.share_factors and self.bias_factor:
        #     dense_features = weights_structure.sum(axis=0)
        #     weights_structure = jax.numpy.concatenate([
        #         xjd.expand_dims(xjd.expand_dims(
        #             dense_features / dense_features.sum(),
        #             0,
        #             weights_structure.shape[0],
        #         ), 0, 1),
        #         xjd.expand_dims(
        #             weights_structure, 0, weights_pca.shape[1] - 1
        #         ),
        #     ])
        else:
            # as this means separate latent_feature * feature weights
            # per factor
            assert weights_structure.shape >= 3

        # n_latents = n_factors, latent_features, n_features

        weights_pca_agg = jax.numpy.multiply(
            # features * n_latent_features, n_factors
            xjd.expand_dims(weights_pca, 1, latents.shape[1]),
            jax.numpy.transpose(weights_structure, (2, 1, 0,))
            #
        ).sum(axis=0).T
        #  latent_features, factors

        return weights_pca_agg

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:

        latents = self.latents.access(state)

        weights_pca = xt.ituple(self.weights_pca.access(state))
        weights_structure = self.weights_structure.access(state)

        res = weights_pca.map(
            functools.partial(self.f_apply, latents=latents),
            weights_structure,
        )

        # alternative is don't fit latents at all
        # and just minimise the variance per latent_feature per factor

        w_stack = jax.numpy.vstack(res.pipe(list))

        # latent_mu = w_stack.mean(axis=0)
        # latent_var = jax.numpy.var(w_stack, axis=0)
        
        return res, jax.numpy.stack(res.map(
            lambda v: jax.numpy.abs(v - latents).mean()
        ).pipe(list)).mean()

# ---------------------------------------------------------------
