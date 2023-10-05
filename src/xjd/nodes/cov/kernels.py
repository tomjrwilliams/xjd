
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

# NOTE: calc as classmethod so can also have a full gp operator that also does the sampling, without re-implementing the kernel 

# for the below, way to include vmap in the same class definition?
# or just have V_GP_... - probably simpler to do that.


# TODO: where vector valued, just take the sum over the diff per dim


@xt.nTuple.decorate(init=xjd.init_null)
class Kernel_Sum(typing.NamedTuple):

    kernels: xt.iTuple = xt.iTuple()
    kernel: typing.Optional[xjd.Loc] = None

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Kernel_Sum, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        if len(self.kernels):
            kernels = self.kernels.map(
                lambda k: k.access(state)
            ).map(
                lambda v: (
                    v if isinstance(v, xt.iTuple) else xt.iTuple((v,))
                )
            ).flatten()
        else:
            kernels = xt.iTuple()
        if self.kernel is not None:
            kernels = kernels.extend(self.kernel.access(state))
        agg = jax.numpy.stack(list(kernels))
        return agg.sum(axis=0)
    


@xt.nTuple.decorate(init=xjd.init_null)
class Kernel_Product(typing.NamedTuple):

    # take others as fields
    # but hence assume that sites are compatible

    c: float

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Kernel_Product, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self
    

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xjd.init_null)
class Kernel_Constant(typing.NamedTuple):

    c: float

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Kernel_Constant, tuple, xjd.SiteValue]: ...
    
    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self
    

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xjd.init_null)
class Kernel_Linear(typing.NamedTuple):

    a: xjd.Loc
    c: xjd.Loc
    sigma: xjd.Loc
    data: xjd.Loc

    sigma_sq: bool = True

    # assumed 1D

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Kernel_Linear, tuple, xjd.SiteValue]: ...
    
    @classmethod
    def f(cls, data, a, c, sigma, sigma_sq = True):

        diffs = data - c
        diffs_ = xjd.expand_dims(diffs, 0, 1)
        # 1, n_points

        # scale

        return (
            (diffs_ * diffs_.T) * (
                jax.numpy.square(sigma)
                if sigma_sq
                else sigma
            )
        ) + jax.numpy.square(a)

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        a = self.a.access(state)
        c = self.c.access(state)
        sigma = self.sigma.access(state)
        data = self.data.access(state)
        return self.f(data, a, c, sigma, sigma_sq=self.sigma_sq)


@xt.nTuple.decorate(init=xjd.init_null)
class VKernel_Linear(typing.NamedTuple):

    a: xjd.Loc
    c: xjd.Loc
    sigma: xjd.Loc
    data: xjd.Loc

    sigma_sq: bool = True

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[VKernel_Linear, tuple, xjd.SiteValue]: ...

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        a = self.a.access(state)
        c = self.c.access(state)
        sigma = self.sigma.access(state)
        data = self.data.access(state)
        return xt.iTuple(a).zip(c, sigma).mapstar(
            lambda _a, _c, _sigma: Kernel_Linear.f(data, _a, _c, _sigma, sigma_sq= self.sigma_sq)
        )


@xt.nTuple.decorate(init=xjd.init_null)
class Kernel_VLinear(typing.NamedTuple):

    a: xjd.Loc
    c: xjd.Loc
    sigma: xjd.Loc
    data: xjd.Loc

    sigma_sq: bool = True

    # assumed 1D

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Kernel_VLinear, tuple, xjd.SiteValue]: ...
    
    @classmethod
    def f(cls, data, a, c, sigma, sigma_sq = True):

        diffs = (xjd.expand_dims(data, 0, 1) - (
            xjd.expand_dims(c, 0, 1).T
        )).T
        # n_points, n_c

        diffs_ = xjd.expand_dims(diffs, 0, diffs.shape[0])

        diff_prod = (
            diffs_ * jax.numpy.transpose(
                diffs_, (1, 0, 2),
            )
        )

        diff_soft = jax.nn.softmax(diff_prod, axis = 2)
        diff_weights = 1 - diff_soft

        return (
            # diff_prod.mean(axis=-1) * (
            (diff_weights * diff_prod).sum(axis=-1) * (
                jax.numpy.square(sigma)
                if sigma_sq
                else sigma
            )
        ) + jax.numpy.square(a)

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        a = self.a.access(state)
        c = self.c.access(state)
        sigma = self.sigma.access(state)
        data = self.data.access(state)
        return self.f(data, a, c, sigma, sigma_sq=self.sigma_sq)


@xt.nTuple.decorate(init=xjd.init_null)
class VKernel_VLinear(typing.NamedTuple):

    a: xjd.Loc
    c: xjd.Loc
    sigma: xjd.Loc
    data: xjd.Loc

    sigma_sq: bool = True

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[VKernel_VLinear, tuple, xjd.SiteValue]: ...

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        a = self.a.access(state)
        c = self.c.access(state)
        sigma = self.sigma.access(state)
        data = self.data.access(state)
        return xt.iTuple(a).zip(c, sigma).mapstar(
            lambda _a, _c, _sigma: Kernel_VLinear.f(data, _a, _c, _sigma, sigma_sq = self.sigma_sq)
        )
       
# ---------------------------------------------------------------

small = 10 ** -4

# aks squared exponential
@xt.nTuple.decorate(init=xjd.init_null)
class Kernel_RadialBasisFunction(typing.NamedTuple):

    # NOTE: https://www.cs.toronto.edu/~duvenaud/cookbook/

    sigma: xjd.Loc
    l: xjd.Loc
    data: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Kernel_RadialBasisFunction, tuple, xjd.SiteValue]: ...

    @classmethod
    def f(cls, data, sigma, l):
        return utils.funcs.kernel_rbf(
            utils.funcs.diffs_1d(data), sigma=sigma, l=l
        )

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        sigma = self.sigma.access(state)
        data = self.data.access(state)
        l = self.l.access(state)
        return self.f(data, sigma, l)

Kernel_RBF = Kernel_RadialBasisFunction
Kernel_SE = Kernel_RBF
Kernel_SquaredExponential = Kernel_SE

# ---------------------------------------------------------------
   

@xt.nTuple.decorate(init=xjd.init_null)
class Kernel_RationalQuadratic(typing.NamedTuple):

    # NOTE: https://www.cs.toronto.edu/~duvenaud/cookbook/

    # NOTE: as a -> inf, RQ -> RBF

    sigma: xjd.Loc
    l: xjd.Loc
    a: xjd.Loc
    data: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Kernel_RationalQuadratic, tuple, xjd.SiteValue]: ...

    @classmethod
    def f(cls, data, sigma, l, a):
        return utils.funcs.kernel_rq(
            utils.funcs.diffs_1d(data),
            sigma=sigma,
            l = l,
            a = a,
        )

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        sigma = self.sigma.access(state)
        data = self.data.access(state)
        l = self.l.access(state)
        a = self.a.access(state)
        return self.f(data, sigma, l, a)



Kernel_RQ = Kernel_RationalQuadratic

# ---------------------------------------------------------------
   

@xt.nTuple.decorate(init=xjd.init_null)
class Kernel_Gaussian(typing.NamedTuple):

    # NOTE: https://francisbach.com/cursed-kernels/

    sigma: xjd.Loc
    data: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Kernel_Gaussian, tuple, xjd.SiteValue]: ...

    @classmethod
    def f(cls, data, sigma):
        return utils.funcs.kernel_gaussian(
            utils.funcs.diffs_1d(data), sigma=sigma
        )

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        sigma = self.sigma.access(state)
        data = self.data.access(state)
        return self.f(data, sigma)

# ---------------------------------------------------------------
   

@xt.nTuple.decorate(init=xjd.init_null)
class Kernel_Exponential(typing.NamedTuple):

    # NOTE: https://francisbach.com/cursed-kernels/

    sigma: xjd.Loc
    data: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Kernel_Exponential, tuple, xjd.SiteValue]: ...

    @classmethod
    def f(cls, data, sigma):
        return utils.funcs.kernel_exponential(
            utils.funcs.diffs_1d(data), sigma = sigma
        )

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        sigma = self.sigma.access(state)
        data = self.data.access(state)
        return self.f(data, sigma)

# ---------------------------------------------------------------
   

@xt.nTuple.decorate(init=xjd.init_null)
class Kernel_Laplacian(typing.NamedTuple):

    sigma: xjd.Loc
    data: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Kernel_Exponential, tuple, xjd.SiteValue]: ...

    @classmethod
    def f(cls, data, sigma):
        return utils.funcs.kernel_laplacian(
            utils.funcs.diffs_1d(data), sigma=sigma
        )

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        sigma = self.sigma.access(state)
        data = self.data.access(state)
        return self.f(data, sigma)

# ---------------------------------------------------------------
   

@xt.nTuple.decorate(init=xjd.init_null)
class Kernel_Cauchy(typing.NamedTuple):

    # NOTE: https://francisbach.com/cursed-kernels/

    sigma: xjd.Loc
    data: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Kernel_Cauchy, tuple, xjd.SiteValue]: ...

    @classmethod
    def f(cls, data, sigma):
        return utils.funcs.kernel_cauchy(
            utils.funcs.diffs_1d(data), sigma=sigma
        )

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        sigma = self.sigma.access(state)
        data = self.data.access(state)
        return self.f(data, sigma)

# ---------------------------------------------------------------
   

@xt.nTuple.decorate(init=xjd.init_null)
class Kernel_Triangular(typing.NamedTuple):

    # NOTE: https://francisbach.com/cursed-kernels/

    sigma: xjd.Loc
    data: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Kernel_Triangular, tuple, xjd.SiteValue]: ...

    @classmethod
    def f(cls, data, sigma):
        return utils.funcs.kernel_triangular(
            utils.funcs.diffs_1d(data), sigma=sigma
        )

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        sigma = self.sigma.access(state)
        data = self.data.access(state)
        return self.f(data, sigma)


# ---------------------------------------------------------------


# TODO: and scale parameter?
@xt.nTuple.decorate(init=xjd.init_null)
class Kernel_Sigmoid(typing.NamedTuple):

    sigma: xjd.Loc
    data: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Kernel_Sigmoid, tuple, xjd.SiteValue]: ...

    @classmethod
    def f(cls, data, sigma):
        return utils.funcs.kernel_sigmoid(
            utils.funcs.diffs_1d(data), 
            # sigma=sigma
        )

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        sigma = self.sigma.access(state)
        data = self.data.access(state)
        return self.f(data, sigma)


# ---------------------------------------------------------------

# TODO: and scale parameter?
@xt.nTuple.decorate(init=xjd.init_null)
class Kernel_Logistic(typing.NamedTuple):

    sigma: xjd.Loc
    data: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Kernel_Logistic, tuple, xjd.SiteValue]: ...

    @classmethod
    def f(cls, data, sigma):
        return utils.funcs.kernel_logistic(
            utils.funcs.diffs_1d(data), 
            # sigma=sigma
        )

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        sigma = self.sigma.access(state)
        data = self.data.access(state)
        return self.f(data, sigma)

# ---------------------------------------------------------------

        

@xt.nTuple.decorate(init=xjd.init_null)
class Kernel_OrnsteinUhlenbeck(typing.NamedTuple):

    # cov = σ^2 / 2θ * exp(−θ(x+y))(1−exp(2θ(x−y)))
    # theta -> sigma

    sigma: xjd.Loc
    data: xjd.Loc

    def init(
        self, site: xjd.Site, model: xjd.Model, data = None
    ) -> tuple[Kernel_Logistic, tuple, xjd.SiteValue]: ...

    @classmethod
    def f(cls, data, sigma):
        return utils.funcs.kernel_ou(
            utils.funcs.diffs_1d(data), 
            sigma=sigma,
        )

    def apply(
        self,
        site: xjd.Site,
        state: xjd.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        sigma = self.sigma.access(state)
        data = self.data.access(state)
        return self.f(data, sigma)
     
Kernel_OU = Kernel_OrnsteinUhlenbeck

# ---------------------------------------------------------------
