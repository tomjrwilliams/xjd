
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
from .. import params


# ---------------------------------------------------------------

# TODO: inv covar (assuming diagonalised representation)


# NOTE: assumes eigvals already positive constrained
# also eigval (not singular value, so no need to square)
@xt.nTuple.decorate(init=xf.init_null)
class Eigen_Cov(typing.NamedTuple):

    eigvals: xf.Loc
    eigvecs: xf.Loc

    vmax: typing.Optional[float] = None

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Eigen_Cov, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        eigvals = self.eigvals.access(state)
        w = self.eigvecs.access(state)

        if len(eigvals.shape) == 1:
            scale = (
                eigvals * jax.numpy.eye(eigvals.shape[-1])
            )
            cov = jax.numpy.matmul(
                jax.numpy.matmul(w, scale), w.T
            )
        else:
            scale = (
                xf.expand_dims(
                    eigvals, 1, 1
                ) * jax.numpy.eye(eigvals.shape[-1])
            )
            cov = jax.numpy.matmul(
                jax.numpy.matmul(w, scale),
                jax.numpy.transpose(w, (0, 2, 1))
            )

        if self.vmax:
            return jax.numpy.clip(
                cov, a_min=-1 * self.vmax, a_max=self.vmax
            )

        return cov

# ---------------------------------------------------------------
