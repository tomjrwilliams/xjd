
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
from ... import utils

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xjd.init_null)
class EM(typing.NamedTuple):
    
    param: xjd.Location
    optimal: xjd.Location # optimal at this step from em algo

    cut_tree: bool = False

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[EM, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        param = self.param.access(state)
        optimal = self.optimal.access(state)
        return utils.funcs.loss_mse(
            param,
            ( 
                jax.lax.stop_gradient(optimal)
                if self.cut_tree
                else optimal
            )
        )


@xt.nTuple.decorate(init=xjd.init_null)
class EM_MatMul(typing.NamedTuple):
    
    raw: xjd.Location
    optimal: xjd.Location # optimal at this step from em algo

    cut_tree: bool = False

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[EM_MatMul, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        raw = self.raw.access(state)
        optimal = self.optimal.access(state)
        param = jax.numpy.matmul(
            jax.numpy.transpose(raw, (0, 2, 1)),
            raw,
        )
        return utils.funcs.loss_mse(
            param,
            ( 
                jax.lax.stop_gradient(optimal)
                if self.cut_tree
                else optimal
            )
        )

# ---------------------------------------------------------------

