
from __future__ import annotations

import itertools
import functools

import numpy
import pandas

import jax
import jax.numpy
import jax.numpy.linalg

import xtuples as xt

from . import shapes
from . import rand
from . import tests


# ---------------------------------------------------------------

def if_none(v, alt, lazy = False):
    return v if v is not None else (
        alt if not lazy else alt()
    )

def if_none_lazy(v, alt):
    return if_none(v, alt, lazy=True)
    
sq = jax.numpy.square
mm = jax.numpy.matmul
mul = jax.numpy.multiply
exp = jax.numpy.exp

# ---------------------------------------------------------------

def linspace(
    start,
    end,
    step = None,
    steps = None,
):
    if steps is None:
        assert step is not None
        steps = int((end - start) / step) + 1
    return numpy.linspace(start, end, steps)

def intspace(start, end, steps = None):
    return linspace(start, end, steps = steps, step = 1.)

# ---------------------------------------------------------------

def set_signs_to(X, axis, signs, i = 0):
    axis_size = X.shape[axis]
    i_shape = xt.iTuple(enumerate(X.shape))
    slice = jax.numpy.ravel(
        jax.numpy.take(X, numpy.array(list(range(
            axis_size
        ))), axis = axis)
    )[i * axis_size: (i + 1) * axis_size]
    X_signs = jax.numpy.sign(slice)
    scale = mul(X_signs, numpy.array(signs))
    for dim, size in i_shape[axis + 1:]:
        scale = shapes.expand_dims(scale, -1, size)
    for dim, size in i_shape[:axis].reverse():
        scale = shapes.expand_dims(scale, 0, size)
    return mul(X, scale)


# ---------------------------------------------------------------


def loss_mabse(l, r):
    return jax.numpy.abs(jax.numpy.subtract(l, r)).mean()

def loss_mse(l, r, mask = None):
    if mask is None:
        return sq(jax.numpy.subtract(l, r)).mean()
    return sq(mul(jax.numpy.subtract(l, r), mask)).mean()

def loss_sumse(l, r):
    return sq(jax.numpy.subtract(l, r)).sum()

def loss_mse_zero(X1):
    return sq(X1).mean()

@functools.lru_cache(maxsize=4)
def loss_mean_zero(axis):
    def f(X):
        return loss_mse_zero(X.mean(axis=axis))
    return f

# ascending just reverse order of xl and xr
def loss_descending(x):
    order = jax.numpy.flip(jax.numpy.argsort(x))
    x_sort = x[order]
    acc = jax.numpy.cumsum(jax.numpy.flip(x_sort))
    xl = x_sort[..., :-1]
    xr = acc[..., 1:]
    return -1 * jax.numpy.subtract(xl, xr).mean()

def to_diag(v):
    return mul(
        jax.numpy.eye(v.shape[0]), v
    )

def loss_diag(X):
    diag = jax.numpy.diag(X)
    diag = mul(
        jax.numpy.eye(X.shape[-1]), diag
    )
    return loss_mse(X, diag)


def loss_orthonormal(X):
    XXt = mm(X, shapes.transpose(X))
    eye = jax.numpy.eye(XXt.shape[-1])
    return loss_mse(XXt, eye)

def loss_orthogonal(X):
    XXt = mm(X, shapes.transpose(X))
    return loss_diag(XXt)


# https://proceedings.neurips.cc/paper_files/paper/2019/file/7dd0240cd412efde8bc165e864d3644f-Paper.pdf
def loss_eigenvec(cov, w, eigvals):
    cov_w = mm(cov, w)
    w_scale = mul(shapes.expand_dims(eigvals, 0, 1), w)

    _mse = loss_mse(cov_w, w_scale)
    _norm = loss_eigenvec_norm(w, eigvals)

    return _mse + _norm

# NOTE: assumes eigvals already positive constrained
def loss_eigenvec_norm(w, eigvals):
    norm = mm(shapes.transpose(w), w)
    norm_sq = sq(norm)

    # norm_unit = jax.nn.sigmoid(norm)
    # norm_unit = jax.numpy.tanh(norm_sq)
    # norm_unit = (2 / (1 + exp(-norm_sq))) - 1
    norm_unit = jax.numpy.clip(norm_sq, a_max=1.)

    mul = mul(
        norm_unit, 
        1 + (
            jax.numpy.eye(norm.shape[-1]) * -2
        ),
    )

    # if len(eigvals.shape) > 1:
    #     eigsum = eigvals.sum(axis=0)
    # else:
    #     eigsum = eigvals.sum()

    return mm(mul, eigvals).mean()

# ---------------------------------------------------------------

# data = n_points, n_features
def cov(data, exists = None):
    """
    >>> betas = rand.gaussian(shape=(3, 3,))
    >>> noise = rand.gaussian(shape=(100, 3))
    >>> data = mm(noise, betas)
    >>> exists = numpy.ones(data.shape)
    >>> cov_data = cov(data)
    >>> tests.assert_is_close(cov_data, cov(data, exists), atol=.05)
    >>> exists = rand.bernoulli(shape=data.shape, p = 0.9)
    >>> cov_res = cov(data, exists)
    >>> tests.assert_is_close(cov_res, cov_res.T, atol=.01)
    >>> tests.assert_is_close(cov_data, cov_res, atol=.2)
    """
    if exists is None:
        return jax.numpy.cov(
            jax.numpy.transpose(data)
        )
    exists_cross = shapes.expand_dims(
        exists,
        -1,
        data.shape[-1]
    )
    exists_mask = mul(
        exists_cross,
        jax.numpy.transpose(exists_cross, (0, 2, 1))
    )
    n_exists = exists.sum(axis=0)
    mu = jax.numpy.divide(data.sum(axis=0), n_exists)
    mu_diff = shapes.expand_dims(
        data - shapes.expand_dims(mu, 0, data.shape[0]),
        -1,
        data.shape[-1]
    )
    mu_diff_T = jax.numpy.transpose(
        mu_diff, (0, 2, 1)
    )
    mu_diff_prod = mul(mul(
        mu_diff,
        mu_diff_T
    ), exists_mask)
    n_exists_mask = exists_mask.sum(axis=0) - 1
    res = jax.numpy.divide(
        mu_diff_prod.sum(axis=0),
        n_exists_mask
    )
    return res

# ---------------------------------------------------------------

def euclidean(v, axis = None, small = 10 ** -3):
    return jax.numpy.sqrt(
        jax.numpy.sum(sq(v), axis = -1) + small
    )
    # return jax.numpy.linalg.norm(v, ord=2, axis=axis)

def diffs_1d(data):
    data_ = shapes.expand_dims(data, 0, 1)
    return data_ - data_.T

def diff_euclidean(l, r, small = 10 ** -3):
    diffs = jax.numpy.subtract(l, r)
    return euclidean(diffs, small=small)

def diff_mahalanobis(
    data,
    mu,
    cov = None,
    cov_inv = None,
    sqrt=True,
):
    assert cov is not None or cov_inv is not None
    if cov_inv is not None:
        assert cov is None
    else:
        cov_inv = jax.numpy.linalg.inv(cov)
    mu_diff = data - mu

    if len(mu_diff.shape) == 3:
        lhs = jax.numpy.transpose(
            mu_diff,
            (0, 2, 1),
        )
        rhs = mu_diff
    else:
        lhs = mu_diff.T
        rhs = mu_diff

    res = mm(mm(lhs, cov_inv), rhs)
    if sqrt:
        return jax.numpy.sqrt(res)
    return res

# ---------------------------------------------------------------

small = 10 ** -4

def normalisation_gaussian(cov):
    det = jax.numpy.linalg.det(cov)
    rhs = jax.numpy.power(2 * numpy.pi, cov.shape[-1])
    det_rhs = det * rhs
    res = 1 / jax.numpy.sqrt(det_rhs)
    return res

def likelihood_gaussian(data, mu, cov):
    norm = normalisation_gaussian(cov)
    prob_unnorm = exp(
        (-1 / 2) * diff_mahalanobis(data, mu, cov = cov, sqrt=False)
    )
    return prob_unnorm * norm

def log_likelihood_gaussian(data, mu, cov):
    norm = normalisation_gaussian(cov)
    prob_unnorm = (
        (-1 / 2) * diff_mahalanobis(data, mu, cov = cov, sqrt=False)
    )
    return prob_unnorm + jax.numpy.log(norm)

# ---------------------------------------------------------------

# (batch), n_dims

def normalisation_gaussian_diag(eigvals):
    res = 1 / jax.numpy.sqrt(
        (2 * numpy.pi) * eigvals
    )
    # assert not jax.numpy.isnan(res).any()
    # assert (jax.numpy.abs(res) < 10 ** 5).all()
    return res

def likelihood_gaussian_diag(data, mu, eigvals, eigvecs):
    norm = normalisation_gaussian_diag(eigvals)
    y = mm(eigvecs, data - mu)
    y_sq = sq(y)
    # := (batch), n_dims
    un_norm = exp(- y_sq / (2 * eigvals))
    prob = norm * un_norm
    return jax.numpy.product(prob, axis=-1)

def log_likelihood_gaussian_diag(data, mu, eigvals, eigvecs):
    # assert not (eigvals <= 0).any()
    norm = normalisation_gaussian_diag(eigvals)
    # assert not jax.numpy.isnan(norm).any()
    y = mm(eigvecs, data - mu)
    y_sq = sq(y)
    un_norm = exp(- y_sq / (2 * eigvals))
    # assert not jax.numpy.isnan(norm * un_norm).any(), [
    #     eigvals,
    # ]
    log_prob = jax.numpy.log(norm * un_norm)
    # assert not jax.numpy.isnan(log_prob).any()
    return jax.numpy.sum(log_prob, axis=-1)


# ---------------------------------------------------------------

# sigma ie. standard deviation
def alpha_beta_steady_state_kalman_params(
    sigma_process, sigma_noise, T = 1
):
    l = (sigma_process * sq(T)) / sigma_noise
    r = (
        4 + l - jax.numpy.sqrt((8 * l) + sq(l))
    ) / 4
    alpha = 1 - sq(r)
    beta = (2 * (2 - alpha)) - (4 * jax.numpy.sqrt(1 - alpha))
    return alpha, beta

ab_kalman_params = alpha_beta_steady_state_kalman_params

def alpha_beta_steady_state_residual_variance(
    sigma_noise, alpha
):
    return sq(sigma_noise) / (
        1 - sq(alpha)
    )

# ---------------------------------------------------------------

def svm_hyperplane(phi_x, w, b):
    # b = scalar
    # w = weight vector
    # phi(x) = fixed transform of x (eg. kernel)
    return mm(w.T, phi_x) + b

def svm_perp_distance(t, Phi_X, w, b):
    # t = target value, -1 or 1 
    return mul(
        t, svm_hyperplane(Phi_X, w, b)
    ) / euclidean(w)


# multi-class can be one vs rest
# ie. t(0) vs t(1 ...), t(1) vs (2 ...) etc.

# or can design some kind of separability margin
# so particular points are only +1 for one class
# eg. a hinge loss on total +1 predictions per label


# slack >= 0
def svm_constraint(t, slack, Phi_X, w, b):
    y = svm_hyperplane(Phi_X, w, b)
    return mul(t, y) + slack # >= 1


# either set slack to zero (fixed)
# then minimise sq(euclidean(w)) / 2, subject to constraint ^

# or minimise C * sum(slack) + sq(euclidean(w)) / 2
# where as C -> inf, recover the original slack=0

# can write a very similar regression problem
# by defining an error tube around the result
# and loss = zero within the tube
# so the slack is then the error beyond the tube
    

# ---------------------------------------------------------------

def linear(data, a, b):
    return (data * b) + a

def expit(data):
    return jax.scipy.special.expit(data)

def logistic(data):
    return 1 / (
        exp(data) + 2 + exp(-data)
    )

kernel_logistic = logistic

def sigmoid(data):
    return (2 / numpy.pi) * (
        1 / (exp(data) + exp(-data))
    )

kernel_sigmoid = sigmoid

def kernel_cosine(data):
    return (numpy.pi / 4) * (
        jax.numpy.cos(
            (numpy.pi / 2) * expit(data)
        )
    )

def rbf(data, l = 1., sigma = 1.):

    data_sq = sq(data)
    sigma_sq = sq(sigma)

    l_sq_2 = 2 * sq(l)

    return exp(
        -1 * (data_sq / l_sq_2)
    ) * sigma_sq

kernel_rbf = rbf

def rq(data, sigma = 1., l = 1., a = 0.):

    sigma_sq = sq(sigma)
    a_2_ls_sq = 2 * sq(l) * a

    data_sq = sq(data)

    return jax.numpy.power(
        1 + (data_sq / a_2_ls_sq),
        -a
    ) * sigma_sq

kernel_rq = rq

def kernel_gaussian(data, sigma = 1.):

    sigma_sq = sq(sigma)
    data_sq = sq(data)

    norm = 1 / (2 * sigma_sq)

    return exp(
        1 - (norm * data_sq)
    )

def kernel_exponential(data, sigma = 1.):

    data_abs = jax.numpy.abs(data)

    norm = 1 / sigma

    return exp(
        1 - (norm * data_abs)
    )

def laplacian(data, sigma = 1.):

    # because has to be positive
    sigma_sq = sq(sigma)
    data_sq = sq(data)

    return exp(
        - sigma_sq * data_sq
    )

kernel_laplacian = laplacian

def cauchy(data, sigma = 1.):

    sigma_sq = sq(sigma)
    data_sq = sq(data)

    return 1 / (
        1 + (data_sq / sigma_sq)
    )

kernel_cauchy = cauchy

def triangular(data, sigma = 1.):
    
    data_abs = jax.numpy.abs(data)

    return jax.numpy.clip(
        1 - (data_abs / (2 * sigma)),
        a_min=0.,
    )

kernel_triangular = triangular

def kernel_ou(data, sigma):

    data_sq = sq(data)
    
    return exp(
        (- data_sq) / sigma
    )

# ---------------------------------------------------------------

def sigmoid_curve(x, scale = 1., upper = 1, mid = 0):
    return upper / (
        1 + exp(-1 * ((scale * x) - mid))
    )

def overextension(x, mid = 0):
    return x * exp(
        -1 * sq(x - mid)
    )

def overextension_df(df, mid=0):
    return pandas.DataFrame(
        overextension(df.values),
        columns=df.columns,
        index=df.index,
    )

# rate??
def gaussian(x, rate = 1, mid = 0):
    return 2 / (
        1 + exp(rate * (x - mid).square)
    )

def gaussian_flipped(x, rate = 1, mid = 0):
    return 1 - gaussian(x, rate = rate, mid = mid)

def gaussian_sigmoid(x, rate = 1, mid = 0):
    return 1 + (-1 / (
        1 + exp(-1 * rate * (x - mid).square)
    ))


# TODO: eg. gaussian surface for convolution kernels


def slope(x, rate = 1):
    return jax.numpy.log(1 + exp(rate * x))

def trough(x, mid):
    return 1 / (
        1 + exp(-x (x - mid))
    )


# ---------------------------------------------------------------

# hyperbolic tangent is an s curve

# of a falling object at time t
# is just the positive side, limiting to at terminal velocity
def velocity(
    t,
    mass,
    gravity,
    drag,
    density,
    area,
    g=10,
):
    alpha_square = (
        density * area * density,
    ) / (2 * mass * gravity)
    #
    alpha = alpha_square ** (1/2)
    #
    return (1 / alpha) * jax.numpy.tanh(
        alpha * g * t
    )


# sech(x) looks a bit like a gaussian
# = 2 / (e^x + e^-x)

# tanh(x) s curve
# = (e^x - e^-x ) / (e^x + e^-x )
# = (e^2x - 1) / (e^2x + 1)

# ---------------------------------------------------------------

# t = point on the curve
# l1 ... = decay
# b1 ... = factor embedding
# exponential term = loading, function of t and l
# so we fit the b1 ... = factor paths

def nelson_siegel_1987(t, b1, b2, b3, l):
    l_t = l * t
    exp_l_t = exp(-l_t)
    b2_exp = (1 - exp_l_t) / l_t
    return (
        b1
        + mul(b2, b2_exp)
        + mul(b3, b2_exp - exp_l_t)
    )
    # + error ** tau

nelson_sigel = nelson_siegel_1987

def bliss_1997(t, b1, b2, b3, l1, l2):
    l1_t = l1 * t
    l2_t = l2 * t
    exp_l1_t = exp(-l1_t)
    exp_l2_t = exp(-l2_t)
    b2_exp = (1 - exp_l1_t) / l1_t
    b3_exp = ((1 - exp_l2_t) / l2_t) - exp_l2_t
    return (
        b1
        + mul(b2, b2_exp)
        + mul(b3, b3_exp)
    )
    # + error ** tau

bliss = bliss_1997

def svensson_1994(t, b1, b2, b3, b4, l1, l2):
    l2_t = l2 * t
    exp_l2_t = exp(-l2_t)
    b4_exp = ((1 - exp_l2_t) / l2_t) - exp_l2_t
    return (
        b1 
        + nelson_siegel_1987(t, 0, b2, b3, l1)
        + mul(b4, b4_exp)
    )
    # + error ** tau

svensson = svensson_1994

def five_factor_2011(t, b1, b2, b3, b4, b5, l1, l2):
    return (
        b1 
        + nelson_siegel_1987(t, 0, b2, b4, l1)
        + nelson_siegel_1987(t, 0, b3, b5, l2)
    ) 
    # + error ** tau

five_factor = five_factor_2011

# ---------------------------------------------------------------
