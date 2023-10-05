
import functools
import jax

from . import shapes

# ---------------------------------------------------------------

# context generator for key scope

KEYS: dict = {}

def next_key(seed = 69):
    if KEYS.get(seed, None) is None:
        KEYS[seed] = jax.random.PRNGKey(seed)
    KEYS[seed], subkey = jax.random.split(KEYS[seed])
    return subkey

def next_keys(n, seed = 69):
    if seed not in KEYS:
        KEYS[seed] = jax.random.PRNGKey(seed)
    KEYS[seed], *subkeys = jax.random.split(KEYS[seed], num=n + 1)
    return jax.numpy.vstack(subkeys) # type: ignore

def reset_keys(seed=69):
    KEYS[seed] = jax.random.PRNGKey(seed)
    return

# ---------------------------------------------------------------

def uniform(shape, seed = 69):
    return jax.random.uniform(
        next_key(seed=seed), 
        shape = tuple(shape),
        #
    )
    
def beta(a, b, shape=None, seed = 69):
    return jax.random.beta(
        a, b,
        next_key(seed=seed), 
        shape = tuple(shape),
        #
    )

def bernoulli(p = None, shape = None, seed = 69):
    return jax.random.bernoulli(
        next_key(seed=seed),
        p=p,
        shape=shape,
    )

# ---------------------------------------------------------------

def gaussian(shape=None, mu=None, var=None, seed = 69):
    # if mu and cov one, need shape
    # else can infer shape from mu / cov
    # if shape given, assert matches mu / cov
    return jax.random.normal(
        next_key(seed=seed), 
        shape = tuple(shape),
        #
    )

def v_gaussian(n, shape=None, mu=None, var=None, seed = 69):
    keys = next_keys(n, seed=seed)
    f = jax.vmap(functools.partial(
        jax.random.normal, 
        shape=tuple(shape),
        #
    ))
    return f(keys)

def mv_gaussian(shape=None, mu=None, cov=None, seed = 69):
    if shape is None:
        assert mu is not None or cov is not None
    return jax.random.multivariate_normal(
        next_key(seed=seed), 
        mu,
        cov,
        # shape = tuple(shape),
        #
        method='svd',
    )

def v_mv_gaussian(n, shape = None, mu=None, cov=None, seed = 69):
    keys = next_keys(n, seed=seed)
    f = jax.vmap(functools.partial(
        jax.random.multivariate_normal, 
        # shape=tuple(shape),
        mean=mu,
        cov=cov,
        method='svd',
        #
    ))
    return f(keys)

# # ---------------------------------------------------------------

# actually n batch dims more useful?
def f_norm_l2(v, n_smaple_dims = None):
    v_sq = jax.numpy.square(v)
    scale = shapes.expand_dims_like(
        jax.numpy.sqrt(v_sq.sum(axis = -1)), -1, v
    )
    return jax.numpy.divide(v, scale)

def norm_gaussian(shape, n = 1):
    v = gaussian(shape)
    res = f_norm_l2(v)
    return res

# ---------------------------------------------------------------

# mu can be vector, in which case we get correalted random walk
# summing over the (not necessarily diag cov) samples
def gaussian_walk(n, shape=None, mu=None, var=None, seed=69):
    shape = shape + (n,)
    v = gaussian(shape=shape, mu=mu, var=var, seed=seed)
    return jax.numpy.cumsum(v, axis = -1)

def v_gaussian_walk(n, shape=None, mu=None, var=None, seed=69):
    return

# geometric walk ie. cum prod not cumsum
def gaussian_gwalk(n, shape=None, mu=None, var=None, seed=69):
    return

def v_gaussian_gwalk(n, shape=None, mu=None, var=None, seed=69):
    return

# mu can be vector, in which case we get correalted random walk
# summing over the (not necessarily diag cov) samples
def mv_gaussian_walk(n, shape=None, mu=None, cov=None, seed=69):
    shape = shape + (n,)
    v = mv_gaussian(shape=shape, mu=mu, cov=cov, seed=seed)
    return jax.numpy.cumsum(v, axis = -1)

def v_mv_gaussian_walk(n, shape=None, mu=None, var=None, seed=69):
    return

# geometric walk ie. cum prod not cumsum
def mv_gaussian_gwalk(n, shape=None, mu=None, var=None, seed=69):
    return

def v_mv_gaussian_gwalk(n, shape=None, mu=None, var=None, seed=69):
    return

# ---------------------------------------------------------------

def orthogonal(n, shape=None, seed = 69):
    return jax.random.orthogonal(
        next_key(seed=seed), 
        n,
        **({} if not shape else dict(shape=shape))
        #
    )

def vorthogonal(n, shape=None, seed = 69):
    keys = next_keys(n, seed=seed)
    f = jax.vmap(functools.partial(
        jax.random.orthogonal, 
        **({} if not shape else dict(shape=shape))
        #
    ))
    return f(keys)

# ---------------------------------------------------------------

# def random_sample(n, f, seed = 69):
#     for subkey in random_keys(n, seed=seed):
#         yield f(subkey)

# def random_uniform(shape, n):
#     f = lambda subkey: jax.random.uniform(subkey, shape=shape)
#     yield from random_sample(n, f)

# def random_uniform_indices(shape, n, threshold):
#     for probs in random_uniform(shape, n):
#         mask = probs <= threshold
#         yield mask

# def random_choices(v, shape, n, p = None):
#     f = lambda subkey: jax.random.choice(
#         subkey, v, shape=shape, p=p
#     )
#     yield from random_sample(n, f)

# def random_indices(l, shape, n, p = None):
#     f = lambda subkey: jax.random.choice(
#         subkey, jax.numpy.arange(l), shape=shape, p=p, replace=False
#     )
#     yield from random_sample(n, f)

# def random_beta(shape, n, a, b):
#     f = lambda subkey: jax.random.beta(subkey, a, b, shape = shape)
#     yield from random_sample(n, f)

# def random_normal(shape, n):
#     f = lambda subkey: jax.random.normal(subkey, shape=shape)
#     yield from random_sample(n, f)

# def random_orthogonal(shape, n):
#     f = lambda subkey: jax.random.orthogonal(
#         subkey, 
#         n=1,
#         shape=shape,
#     ).squeeze().squeeze()
#     yield from random_sample(n, f)


# # ---------------------------------------------------------------
