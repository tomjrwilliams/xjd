
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

# NOTE: below is for fitting the parameters
# under a given history of position and velocity
# all at once

# we make predictions for full position history
# so we only take residuals on positions[1:] vs predictions[:-1]

# can optionally extend prediction if a non additive dynamic
# function of next(position) given current(position, velocity)

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xf.init_null)
class Prediction(typing.NamedTuple):
    
    position: xf.Loc
    velocity: xf.Loc

    delta_t: xf.OptionalLoc = None

    # alpha: xf.Loc
    # beta: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Prediction, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        position = self.position.access(state)
        velocity = self.velocity.access(state)
        if self.delta_t is None:
            return jax.numpy.concatenate([
                position[..., :1, :],
                position[..., 1:, :] + velocity
            ])
        delta_t = self.delta_t.access(state)
        return jax.numpy.concatenate([
            position[..., :1, :],
            position[..., 1:, :] + (velocity * delta_t)
        ])
    
@xt.nTuple.decorate(init=xf.init_null)
class Update(typing.NamedTuple):
    
    position: xf.Loc

    prediction: xf.Loc
    velocity: xf.Loc

    alpha: xf.Loc
    beta: xf.Loc

    delta_t: xf.OptionalLoc = None

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Update, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:

        position = self.position.access(state)
        prediction = self.prediction.access(state)
        velocity = self.velocity.access(state)
        
        alpha = self.alpha.access(state)
        beta = self.beta.access(state)

        pred_clip = prediction[..., :-1, :]
        vel_clip = velocity[..., :-1, :]

        # assume same shape as pca: n_days, n_cols
        residual = position - pred_clip

        if self.delta_t is None:
            beta_scale = beta
        else:
            delta_t = self.delta_t.access(state)
            beta_scale = beta / delta_t

        # re-attach final values for which we don't have residual
        # so shape matches
        return (
            jax.numpy.concatenate([
                pred_clip + (alpha * residual),
                prediction[..., -1:, :]
            ]),
            jax.numpy.concatenate([
                vel_clip + (beta_scale * residual[..., 1:, :]),
                velocity[..., -2:-1, :],
            ]),
        )

# a, b > 0
# a, b < 1
# elif a > 1, we amplify the signal
# elif b > 1 (strictly <= 2), we amplify noise

# ---------------------------------------------------------------
