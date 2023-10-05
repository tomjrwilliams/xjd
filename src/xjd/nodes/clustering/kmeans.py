
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


@xt.nTuple.decorate(init=xjd.init_null)
class KMeans_Labels(typing.NamedTuple):
    
    k: int
    mu: xjd.Location
    var: xjd.Location

    data: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[KMeans_Labels, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        # https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
        # https://theory.stanford.edu/~sergei/papers/kMeans-socg.pdf

        mu = self.mu.access(state)
        var = self.var.access(state)

        data = self.data.access(state)

        diffs = jax.numpy.subtract(
            xjd.expand_dims(data, -1, mu.shape[0]),
            xjd.expand_dims(mu.T, 0, data.shape[0]),
        )
        return jax.numpy.argmin(
            jax.numpy.square(diffs).sum(axis=1), axis=1
        )

    
# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xjd.init_null)
class KMeans_EM_MeanDiff(typing.NamedTuple):
    
    k: int

    mu: xjd.Location
    var: xjd.Location
    data: xjd.Location
    labels: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[KMeans_EM_MeanDiff, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        # https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
        # https://theory.stanford.edu/~sergei/papers/kMeans-socg.pdf

        data = self.data.access(state)
        labels = self.labels.access(state)
        # label: n_data

        mu = self.mu.access(state)
        var = self.var.access(state)

        inds = xjd.expand_dims(
            jax.numpy.linspace(
                0, self.k, num=self.k, endpoint=False
            ),
            axis=0,
            size=data.shape[0],
        )

        # very odd behaviour if we try to expand at -1
        labs = xjd.expand_dims(labels, 1, self.k)

        one_hot = jax.numpy.isclose(
            labs,
            inds
        )
        # n_data, n_clusters

        neg_hot = 1 + (-1 * one_hot)

        data_mu = xjd.expand_dims(mu.T, 0, data.shape[0])
        data_exp = xjd.expand_dims(data, -1, self.k,)

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


@xt.nTuple.decorate(init=xjd.init_null)
class KMeans_EM_Naive(typing.NamedTuple):
    
    k: int

    data: xjd.Location
    labels: xjd.Location

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[KMeans_EM_Naive, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        # https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
        # https://theory.stanford.edu/~sergei/papers/kMeans-socg.pdf

        data = self.data.access(state)
        labels = self.labels.access(state)
        # label: n_data

        inds = xjd.expand_dims(
            jax.numpy.linspace(
                0, self.k, num=self.k, endpoint=False
            ),
            axis=0,
            size=data.shape[0],
        )

        # very odd behaviour if we try to expand at -1
        labs = xjd.expand_dims(labels, 1, self.k)

        one_hot = jax.numpy.isclose(
            labs,
            inds
        )
        # n_data, n_clusters
        
        counts = xjd.expand_dims(
            one_hot.sum(axis=0) + 1, 
            axis=0, 
            size=data.shape[1],
        )

        # n_data, n_col, n_clusters

        one_hot = xjd.expand_dims(one_hot, 1, data.shape[1])
        # n_data, (n_col), n_cluster: boolean

        data_labelled = jax.numpy.multiply(
            xjd.expand_dims(data, -1, self.k,),
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
