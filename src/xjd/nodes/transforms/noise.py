
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
from ... import xfactors as xf

# ---------------------------------------------------------------

mm = jax.numpy.matmul

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class Diagonal_Gaussian(typing.NamedTuple):
    
    data: xf.Loc

    eigval: xf.Loc
    eigvec: xf.Loc

    scale: float = 1

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Diagonal_Gaussian, tuple, xf.SiteValue]: ...
    
    # Fx + Bu
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
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

        noise = xf.expand_dims(noise, -1, 1)

        noise = jax.numpy.matmul(
            utils.shapes.transpose(eigvec), 
            noise,
        )

        return data + noise[..., 0]


# ---------------------------------------------------------------

