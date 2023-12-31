
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
class State_Prediction(typing.NamedTuple):
    
    transition: xjd.Loc
    state: xjd.Loc

    noise: xjd.Loc

    input: xjd.OptionalLoc = None
    control: xjd.OptionalLoc = None

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[State_Prediction, tuple, xjd.SiteValue]: ...
    
    # Fx + Bu
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        
        F = self.transition.access(state)
        X = self.state.access(state)

        noise = self.noise.access(state)

        # state: (n_days, n_features)
        # transition = n_features, n_features

        if len(X.shape) < 3:
            X = xjd.expand_dims(X, -1, 1)

        # should now treat state[0] as batch dim
        # and multiply in from the right

        FX = mm(F, X[:-1, :, :])

        # F_inv = jax.numpy.linalg.inv(F)

        if self.input is None:
            res = jax.numpy.concatenate([
                # mm(F_inv, X[1:2, :, :]),
                # X[1:2, :, :],
                X[:1, :, :],
                FX
            ], axis = 0)

            assert res.shape == X.shape, [res.shape, X.shape]

            res = res[..., 0]

            return (
                res,
                res, # + noise,
                res - X[..., 0]
            )

        assert self.control is not None
        B = self.control.access(state)

        U = self.input.access(state)
        U = xjd.expand_dims(U, -1, 1)

        BU = mm(B, U)

        res = jax.numpy.concatenate([
            # mm(F_inv, X[1:2, :, :]),
            # X[1:2, :, :],
            X[:1, :, :],
            FX + BU
        ], axis = 0)[..., 0]
        
        return (
            res,
            res,
            res - X[..., 0]
        )

@xt.nTuple.decorate(init=xjd.init_null)
class Cov_Prediction(typing.NamedTuple):
    
    transition: xjd.Loc
    cov: xjd.Loc
    # ie. previous cov prediction

    noise: xjd.Loc
    # process noise covariance

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Cov_Prediction, tuple, xjd.SiteValue]: ...
    
    # FPFt + Q
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        F = self.transition.access(state)
        P = self.cov.access(state)
        noise = self.noise.access(state)

        # if (jax.numpy.abs(noise) > 10 ** 5).any():
        #     assert False, noise

        assert len(P.shape) == 3, P.shape

        # cov: n_days, n_features, n_features
        # numpy should (?) broadcast lhs dimensions

        P_trim = P[:-1, :, :]

        res = mm(
            mm(F, P_trim), F.T
        )

        return jax.numpy.concatenate([
            # xjd.expand_dims(noise, 0, 1),
            P[:1, :, :],
            res + xjd.expand_dims(noise, 0, P_trim.shape[0])
        ]), P[1:, ...] - res

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xjd.init_null)
class State_Innovation(typing.NamedTuple):
    
    data: xjd.Loc # observations
    observation: xjd.Loc # obs model
    state: xjd.Loc # from the predict step

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[State_Innovation, tuple, xjd.SiteValue]: ...
    
    # z - Hx
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:

        data = self.data.access(state)
        H = self.observation.access(state)
        X = self.state.access(state)

        if len(X.shape) < 3:
            X = xjd.expand_dims(X, -1, 1)

        # assert has batch dim, state.shape[2] == 1
        assert len(X.shape) == 3, X.shape

        data = xjd.expand_dims(data, -1, 1)

        return data - jax.numpy.matmul(H, X)[:-1, :, :]

@xt.nTuple.decorate(init=xjd.init_null)
class Cov_Innovation(typing.NamedTuple):
    
    observation: xjd.Loc
    cov: xjd.Loc # from the predict step
    noise: xjd.Loc # observation noise covariance

    # observation_noise: float = 0.01

    vmax: float = 10.

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Cov_Innovation, tuple, xjd.SiteValue]: ...
    
    # HPHt + R
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        H = self.observation.access(state)
        P = self.cov.access(state)
        noise = self.noise.access(state)

        assert len(P.shape) == 3, P.shape

        # if (jax.numpy.abs(noise) > 10 ** 5).any():
        #     assert False, noise

        # if self.observation_noise:
        #     key = site.loc.random().access(
        #         state, into=jax.numpy.ndarray
        #     )
        #     H = H + (
        #         jax.random.normal(key, shape=H.shape) * self.observation_noise
        #     )

        res = mm(
            H, mm(P, H.T)
        ) + xjd.expand_dims(noise, 0, P.shape[0])

        # if (jax.numpy.abs(res) > 10 ** 5).any():
        #     assert False, dict(
        #         res=res[:10],
        #         P=P[:10],
        #         H=H,
        #     )

        return res

# ---------------------------------------------------------------

small = 10 ** -4


@xt.nTuple.decorate(init=xjd.init_null)
class Kalman_Gain(typing.NamedTuple):
    
    cov: xjd.Loc # prediction
    observation: xjd.Loc
    cov_innovation: xjd.Loc

    # observation_noise: float = 0.01

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Kalman_Gain, tuple, xjd.SiteValue]: ...
    
    # PHtS-1
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        P = self.cov.access(state)
        H = self.observation.access(state)
        cov_innovation = self.cov_innovation.access(state)

        assert len(P.shape) == 3, P.shape

        # if self.observation_noise:
        #     key = site.loc.random().access(
        #         state, into=jax.numpy.ndarray
        #     )
        #     H = H + (
        #         jax.random.normal(key, shape=H.shape) * self.observation_noise
        #     )
        try:
            inv_innovation = jax.numpy.linalg.inv(
                cov_innovation
            )
        except:
            assert False, cov_innovation[:10]
            #  + (
            #     jax.numpy.eye(
            #         cov_innovation.shape[-1]
            #     ) * small
            #     #
            # )
        # )
        # assert not jax.numpy.isnan(inv_innovation).any()

        return mm(
            mm(P, H.T), 
            inv_innovation,
        )

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xjd.init_null)
class State_Updated(typing.NamedTuple):
    
    state: xjd.Loc # pred
    kalman_gain: xjd.Loc
    state_innovation: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[State_Updated, tuple, xjd.SiteValue]: ...
    
    # x + Ky
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        kalman_gain = self.kalman_gain.access(state)
        state_innovation = self.state_innovation.access(state)

        X = self.state.access(state)

        if len(X.shape) < 3:
            X = xjd.expand_dims(X, -1, 1)

        return jax.numpy.concatenate([
            (X[:-1, :, :] + mm(
                kalman_gain[:-1, :, :], state_innovation
            )),
            X[-1:, :, :],
        ], axis = 0)[..., 0]


@xt.nTuple.decorate(init=xjd.init_null)
class Cov_Updated(typing.NamedTuple):
    
    kalman_gain: xjd.Loc
    observation: xjd.Loc
    cov: xjd.Loc # predicted

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Cov_Updated, tuple, xjd.SiteValue]: ...
    
    # (I - KH)P
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        kalman_gain = self.kalman_gain.access(state)
        H = self.observation.access(state)

        P = self.cov.access(state)
        # P_trim = P[:-1, :, :]
        
        I = xjd.expand_dims(
            jax.numpy.eye(P.shape[1]), 0, P.shape[0]
        )

        return mm(I - mm(kalman_gain, H), P)
        # jax.numpy.concatenate([
        #     mm(I - mm(kalman_gain, H), P),
        #     P[-1:, :, :],
        # ], axis = 0)


@xt.nTuple.decorate(init=xjd.init_null)
class Residual(typing.NamedTuple):
    
    data: xjd.Loc
    observation: xjd.Loc
    state: xjd.Loc # post update

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Residual, tuple, xjd.SiteValue]: ...
    
    # z - Hx
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)

        H = self.observation.access(state)
        X = self.state.access(state)

        # because we compress above so we cna feed back
        # with markov in the same shape as input
        # ie. without rhs [1] dim (so n_days goes to batch-dim)

        X = xjd.expand_dims(X, -1, 1)
        data = xjd.expand_dims(data, -1, 1)
        
        return data - mm(H, X)[:-1, :, :]


# ---------------------------------------------------------------

# F = state transition
# H = observation model

# Q = covariance (true) state process noise
# R = covariance of observation noise

# (optional)
# u = control input vector
# B = control-input 

# w = process noise
# w = N(0, Q)

# x = true state
# x = Fx + Bu + w

# v = observation noise
# v = N(0, R)

# z = observation
# z = Hx + v

# NOTE: so we have a covariance of the state locations
# per time step: it's a running uncertainty gauge

# ---------------------------------------------------------------

# if non linear function, use extended:

# x = f(x) ... 
# z = h(x) ... 

# where these are differentiable functions
# use jacobian instead of gradient to update covariance

# unscented:

# batch wise sample around current mean instead of point estimate

# given extended assumes we can linearise the function around 
# the current estimates

# ---------------------------------------------------------------

# eg. state = loc and velocity of factors in pca model

# loc = prev(loc) + prev(velocity) + noise
# velocity = velocity + noise

# possibly velocity is mean reverting w.r.t. loc

# then observation is in return space
# via summing up loc * factor weights over tickers

# ---------------------------------------------------------------
