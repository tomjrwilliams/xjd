
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
class Diagonal_Gaussian(typing.NamedTuple):
    
    data: xjd.Loc

    eigval: xjd.Loc
    eigvec: xjd.Loc

    scale: float = 1

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Diagonal_Gaussian, tuple, xjd.SiteValue]: ...
    
    # Fx + Bu
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        
        data = self.data.access(state)
        eigvec = self.eigvec.access(state)
        eigval = self.eigval.access(state)

        assert site.loc is not None
        key = site.loc.random().access(
            state, into=jax.numpy.ndarray
        )
        noise = (
            jax.random.normal(key, shape=(
                data.shape[0], eigval.shape[-1]
            )) * self.scale
        )
        noise = noise * jax.numpy.sqrt(eigval)

        noise = xjd.expand_dims(noise, -1, 1)

        noise = jax.numpy.matmul(
            utils.shapes.transpose(eigvec), 
            noise,
        )

        return data + noise[..., 0]


# ---------------------------------------------------------------

