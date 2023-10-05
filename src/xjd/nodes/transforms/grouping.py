
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

# below is eg. for latent gp model
# so we first group by (say) sector
# then for each, vmap gp
# then flatten back down & apply constraints (mse, ...)

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class Group_By(typing.NamedTuple):
    
    values: xt.iTuple
    keys: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Group_By, tuple, xf.SiteValue]: ...
    
    # return tuple of values vmapped over indices
    # given by the values in the map(get_location(site_keys))

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self


# ---------------------------------------------------------------
