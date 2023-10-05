
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

from . import vanilla

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class Parametric_Factor(typing.NamedTuple):
    
    features: xf.Location
    params: xf.Location
    # or an operator for the function, which can have its own params site? probably that

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Parametric_Factor, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:

        # given a feature matrix 
        # eg. simple one is a constant column per tenor (of the tenor represented as a float)
        # for rates parametric factor pca
        
        # how to control if we have to apply to a whole matrix or just a vector?
        # eg the tenor can be a single global vector

        # then we just have to scale it each day by the latent factor for the intensity of that factor


        # can have separate vector wise parametric functions
        # that we have a stack operator to join into singl eloadings matrix

        # and then that can be multiplied to the latent factor path
        # latent as opposed to encoded
        # -> yields


        # apply the functions callables loading into the tuple

        return ()