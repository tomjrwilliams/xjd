
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

# ---------------------------------------------------------------

small = 10 ** -4


@xt.nTuple.decorate(init=xf.init_null)
class Eigenvec(typing.NamedTuple):
    
    cov: xf.Location
    weights: xf.Location
    eigvals: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Eigenvec, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        weights = self.weights.access(state)
        cov = self.cov.access(state)
        eigvals = self.eigvals.access(state) + small
        if self.T:
            weights = weights.T
        return utils.funcs.loss_eigenvec(cov, weights, eigvals)

@xt.nTuple.decorate(init=xf.init_null)
class Eigenvec_Cov(typing.NamedTuple):
    
    eigvals: xf.Location
    weights: xf.Location

    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Eigenvec, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        weights = self.weights.access(state)
        eigvals = self.eigvals.access(state) + small
        if self.T:
            weights = weights.T
        if len(weights.shape) == 3:
            eigvals = xf.expand_dims(eigvals, -1, 1)
        return utils.funcs.loss_eigenvec_norm(
            weights, eigvals
        )

@xt.nTuple.decorate(init=xf.init_null)
class Orthonormal(typing.NamedTuple):
    
    data: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Orthonormal, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        X = self.data.access(state)
        if self.T:
            X = X.T
        return utils.funcs.loss_orthonormal(X)


@xt.nTuple.decorate(init=xf.init_null)
class Orthogonal(typing.NamedTuple):
    
    data: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Orthogonal, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        X = self.data.access(state)
        if self.T:
            X = X.T
        return utils.funcs.loss_orthogonal(X)


@xt.nTuple.decorate(init=xf.init_null)
class VEigenvec(typing.NamedTuple):
    
    cov: xf.Location
    weights: xf.Location
    eigvals: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[VEigenvec, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        weights = self.weights.access(state)
        cov = self.cov.access(state)
        eigvals = self.eigvals.access(state)
        if self.T:
            return jax.numpy.vstack([
                xt.iTuple(cov).map(
                    utils.funcs.loss_eigenvec, 
                    xt.iTuple(weights).map(
                        lambda v: v.T
                    ),
                    eigvals,
                ).pipe(list)
            ]).sum()    
        return jax.numpy.vstack([
            xt.iTuple(cov).map(
                utils.funcs.loss_eigenvec, weights, eigvals
            ).pipe(list)
        ]).sum()

@xt.nTuple.decorate(init=xf.init_null)
class VOrthogonal(typing.NamedTuple):
    
    data: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[VOrthogonal, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        X = self.data.access(state)
        if self.T:
            return jax.numpy.vstack([
                utils.funcs.loss_orthogonal(x.T) for x in X
            ]).sum()    
        return jax.numpy.vstack([
            utils.funcs.loss_orthogonal(x) for x in X
        ]).sum()


@xt.nTuple.decorate(init=xf.init_null)
class VOrthonormal(typing.NamedTuple):
    
    data: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[VOrthonormal, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        X = self.data.access(state)
        if self.T:
            return jax.numpy.vstack([
                utils.funcs.loss_orthonormal(x.T) for x in X
            ]).sum()    
        return jax.numpy.vstack([
            utils.funcs.loss_orthonormal(x) for x in X
        ]).sum()
        # return jax.vmap(utils.funcs.loss_orthonormal)(X).sum()


@xt.nTuple.decorate(init=xf.init_null)
class Diagonal(typing.NamedTuple):
    
    data: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[VDiagonal, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        X = self.data.access(state)
        if self.T:
            X = X.T
        return utils.funcs.loss_diag(X)

@xt.nTuple.decorate(init=xf.init_null)
class VDiagonal(typing.NamedTuple):
    
    data: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[VDiagonal, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        X = self.data.access(state)
        if self.T:
            X = X.T
        return jax.vmap(utils.funcs.loss_diag)(X).sum()

@xt.nTuple.decorate(init=xf.init_null)
class WtSW(typing.NamedTuple):
    
    W: xf.Loc # eigvec
    S: xf.Loc # cov

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[WtSW, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        
        W = self.W.access(state)
        S = self.S.access(state)

        res = jax.numpy.matmul(jax.numpy.matmul(W.T, S), W)

        return res.sum() + jax.numpy.square(jax.numpy.clip(
            res, a_max=0.
        )).sum()

@xt.nTuple.decorate(init=xf.init_null)
class XXt_Cov(typing.NamedTuple):
    
    X: xf.Location
    cov: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[XXt_Cov, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        
        X = self.X.access(state)
        cov = self.cov.access(state)

        XXt = jax.numpy.matmul(X, X.T)

        return utils.funcs.loss_mse(XXt, cov)

@xt.nTuple.decorate(init=xf.init_null)
class XD2Xt_Cov(typing.NamedTuple):
    
    X: xf.Location
    D: xf.Location
    cov: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[XD2Xt_Cov, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        
        X = self.X.access(state)
        D = self.D.access(state)
        cov = self.cov.access(state)

        D_sq = jax.numpy.square(D)

        XXt = jax.numpy.matmul(
            jax.numpy.multiply(X, D_sq), X.T
        )

        return utils.funcs.loss_mse(XXt, cov)

@xt.nTuple.decorate(init=xf.init_null)
class EigenVLike(typing.NamedTuple):
    
    weights: xf.Location
    factors: xf.Location

    eigval_max: bool = True

    n_check: typing.Optional[int] = None

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[EigenVLike, tuple, xf.SiteValue]: ...

    @classmethod
    def f_apply(cls, w, f, eigval_max=True):
        
        cov = jax.numpy.cov(f.T)
        eigvals = jax.numpy.diag(cov)

        res = (
            + utils.funcs.loss_descending(eigvals)
            + utils.funcs.loss_orthonormal(w.T)
            + utils.funcs.loss_mean_zero(0)(f)
            + utils.funcs.loss_diag(cov)
        )
        if eigval_max:
            return res + (
                - jax.numpy.sum(jax.numpy.log(1 + eigvals))
                # ridge penalty to counteract eigval max
            )
        return res

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:

        w = self.weights.access(state)
        f = self.factors.access(state)

        return self.f_apply(w, f, eigval_max=self.eigval_max)


@xt.nTuple.decorate(init=xf.init_null)
class VEigenVLike(typing.NamedTuple):
    
    weights: xf.Location
    factors: xf.Location

    eigval_max: bool = True

    n_check: typing.Optional[int] = None

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[VEigenVLike, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:

        w = xt.iTuple(self.weights.access(state))
        f = self.factors.access(state)

        return jax.numpy.stack(w.map(
            functools.partial(
                EigenVLike.f_apply,
                eigval_max=self.eigval_max
            ),
            f,
        ).pipe(list)).sum()

def l1_diag_loss(v):
    return jax.numpy.abs(jax.numpy.diag(v)).mean()

@xt.nTuple.decorate(init=xf.init_null)
class L1_MM_Diag(typing.NamedTuple):
    
    raw: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[L1_MM_Diag, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        raw = self.raw.access(state)
        data = jax.numpy.matmul(
            jax.numpy.transpose(raw, (0, 2, 1)),
            raw,
        )
        return jax.vmap(l1_diag_loss)(data).mean()

# ---------------------------------------------------------------



@xt.nTuple.decorate(init=xf.init_null)
class KernelVsCov(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[KernelVsCov, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        # param weights
        # y_pred = weights . normal
        # mse(y_pred, y)
        # 
        # cov_weights = weights.weights
        # cov_kernel = kernel(latent(x))
        # mse(cov_weights, cov_kernel)
        assert False, self


# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class MinimiseMMSpread(typing.NamedTuple):
    
    data: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[MinimiseMMSpread, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        if self.T:
            data = data.T
        cov = jax.numpy.matmul(
            jax.numpy.transpose(data, (0, 2, 1)),
            data,
        )
        mu = jax.numpy.mean(cov, axis = 0)
        delta = jax.numpy.square(jax.numpy.subtract(
            cov,
            xf.expand_dims(mu, 0, cov.shape[0])
        )).mean()
        return delta



@xt.nTuple.decorate(init=xf.init_null)
class MinimiseVariance(typing.NamedTuple):
    
    data: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[MinimiseVariance, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        if self.T:
            data = data.T
        var = jax.numpy.var(data.flatten())
        return var


@xt.nTuple.decorate(init=xf.init_null)
class MinimiseZSpread(typing.NamedTuple):
    
    data: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[MinimiseZSpread, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        if self.T:
            data = data.T
        var = jax.numpy.var(data.flatten())
        sigma = jax.numpy.sqrt(var)
        mu = jax.numpy.mean(data)
        delta = (data - mu) / sigma
        return jax.numpy.square(delta).mean()


@xt.nTuple.decorate(init=xf.init_null)
class MaxSpread(typing.NamedTuple):
    
    data: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[MaxSpread, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        if self.T:
            data = data.T
        data = xf.expand_dims(data, 0, data.shape[0])
        dataT = jax.numpy.transpose(
            data, (1, 0, 2,)
        )
        delta = jax.numpy.abs(
            jax.numpy.subtract(data, dataT)
        ).sum(axis=-1)
        return -1 * delta.mean()

# ---------------------------------------------------------------

