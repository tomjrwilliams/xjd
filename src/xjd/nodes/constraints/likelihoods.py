
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

from ... import utils
from ... import xjd

# ---------------------------------------------------------------

mm = jax.numpy.matmul

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xjd.init_null)
class Neg_Gaussian(typing.NamedTuple):
    
    data: xjd.Loc
    cov: xjd.Loc

    T: bool = False

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Neg_Gaussian, tuple, xjd.SiteValue]: ...

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        
        X = self.data.access(state)

        cov = self.cov.access(state)

        # eigvals =self.eigvals.access(state)
        # eigvecs =self.eigvecs.access(state)

        if len(X.shape) < 3:
            X = xjd.expand_dims(X, -1, 1)

        if len(cov.shape) < 3:
            cov = xjd.expand_dims(cov, 0, X.shape[0])

        likelihood = utils.funcs.likelihood_gaussian(
            data=X,
            mu=jax.numpy.zeros(
                X.shape,
            ),
            cov=cov
            # [1:, :, :],
        ).mean()

        # log_likelihood = utils.funcs.log_likelihood_gaussian_diag(
        #     data=X[1:, :, :],
        #     mu=FX,
        #     eigvals=xjd.expand_dims(
        #         eigval[1:, ...], -1, 1
        #     ),
        #     eigvecs=eigvec[1:, ...],
        # ).mean()

        return -1 * likelihood


# ---------------------------------------------------------------

