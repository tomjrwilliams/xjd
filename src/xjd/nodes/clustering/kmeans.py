
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

# from jax.config import config 
# config.update("jax_debug_nans", True) 

def reindex_labels(labels):
    order = xt.iTuple(sorted(
        set(labels),
        key=labels.index
    ))
    labels_ordered = {
        label: i for i, label in order.enumerate()
    }
    return labels.map(lambda l: labels_ordered[l]), order

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class KMeans_Labels(typing.NamedTuple):
    
    k: int
    mu: xf.Location
    var: xf.Location

    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[KMeans_Labels, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        # https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
        # https://theory.stanford.edu/~sergei/papers/kMeans-socg.pdf

        mu = self.mu.access(state)
        var = self.var.access(state)

        data = self.data.access(state)

        diffs = jax.numpy.subtract(
            xf.expand_dims(data, -1, mu.shape[0]),
            xf.expand_dims(mu.T, 0, data.shape[0]),
        )
        return jax.numpy.argmin(
            jax.numpy.square(diffs).sum(axis=1), axis=1
        )

    
# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class KMeans_EM_MeanDiff(typing.NamedTuple):
    
    k: int

    mu: xf.Location
    var: xf.Location
    data: xf.Location
    labels: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[KMeans_EM_MeanDiff, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        # https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
        # https://theory.stanford.edu/~sergei/papers/kMeans-socg.pdf

        data = self.data.access(state)
        labels = self.labels.access(state)
        # label: n_data

        mu = self.mu.access(state)
        var = self.var.access(state)

        inds = xf.expand_dims(
            jax.numpy.linspace(
                0, self.k, num=self.k, endpoint=False
            ),
            axis=0,
            size=data.shape[0],
        )

        # very odd behaviour if we try to expand at -1
        labs = xf.expand_dims(labels, 1, self.k)

        one_hot = jax.numpy.isclose(
            labs,
            inds
        )
        # n_data, n_clusters

        neg_hot = 1 + (-1 * one_hot)

        data_mu = xf.expand_dims(mu.T, 0, data.shape[0])
        data_exp = xf.expand_dims(data, -1, self.k,)

        delta_mu_diff = jax.numpy.subtract(data_exp, data_mu)

        self_diff = jax.numpy.multiply(
            jax.numpy.square(delta_mu_diff).sum(axis=1), one_hot
        ).sum(axis=1)
        # n_data diff to self

        other_diff = jax.numpy.multiply(
            jax.numpy.abs(delta_mu_diff).sum(axis=1), neg_hot
        ).mean(axis=1)

        return self_diff - other_diff

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class KMeans_EM_Naive(typing.NamedTuple):
    
    k: int

    data: xf.Location
    labels: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[KMeans_EM_Naive, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        # https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
        # https://theory.stanford.edu/~sergei/papers/kMeans-socg.pdf

        data = self.data.access(state)
        labels = self.labels.access(state)
        # label: n_data

        inds = xf.expand_dims(
            jax.numpy.linspace(
                0, self.k, num=self.k, endpoint=False
            ),
            axis=0,
            size=data.shape[0],
        )

        # very odd behaviour if we try to expand at -1
        labs = xf.expand_dims(labels, 1, self.k)

        one_hot = jax.numpy.isclose(
            labs,
            inds
        )
        # n_data, n_clusters
        
        counts = xf.expand_dims(
            one_hot.sum(axis=0) + 1, 
            axis=0, 
            size=data.shape[1],
        )

        # n_data, n_col, n_clusters

        one_hot = xf.expand_dims(one_hot, 1, data.shape[1])
        # n_data, (n_col), n_cluster: boolean

        data_labelled = jax.numpy.multiply(
            xf.expand_dims(data, -1, self.k,),
            one_hot,
        )

        mu = jax.numpy.divide(
            data_labelled.sum(axis=0),
            counts
        )
        # we manually take the average over count of non zero
        # (n_data), n_col, n_cluster

        # manually scale back up the average
        # and then back down by the non zero values
        var = jax.numpy.divide(
            jax.numpy.var(data_labelled, axis=0) * data.shape[0],
            counts
        )

        return mu.T, var.T


# ---------------------------------------------------------------
