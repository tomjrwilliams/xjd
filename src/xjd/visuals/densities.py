
import numpy
import pandas
import scipy.stats

def clip(x, clip_quantile=None):
    if clip_quantile is None:
        pass
    else:
        if isinstance(clip_quantile, tuple):
            lq, rq = clip_quantile
        else:
            lq = clip_quantile
            rq = 1 - lq
        l = numpy.quantile(x, lq)
        r = numpy.quantile(x, rq)
        x = numpy.clip(x, l, r)
    return x


def gaussian_kde_1d(x, clip_quantile=None):

    x = clip(x, clip_quantile=clip_quantile)

    xmin = x.min()
    xmax = x.max()

    X = numpy.mgrid[xmin:xmax:100j] # type: ignore

    positions = X.ravel()
    kernel = scipy.stats.gaussian_kde(x)
    Z = numpy.reshape(kernel(positions).T, X.shape)
    # z is density over the grid (xmin, xmax)
    return positions, Z

def gaussian_kde_1d_df(xs, key = "key", **kde_kwargs):
    ks = []
    positions = []
    densities = []
    for k, x in xs.items():
        ps, ds = gaussian_kde_1d(x, **kde_kwargs)
        positions.extend(ps)
        densities.extend(ds)
        ks.extend([k for _ in ds])
    return pandas.DataFrame({
        key: ks,
        "position": positions,
        "density": densities,
    })

def gaussian_kde_2d(x, y, clip_quantile = None, n = 30, ravel = True):

    x = clip(x, clip_quantile=clip_quantile)
    y = clip(y, clip_quantile=clip_quantile)

    xmin = x.min()
    xmax = x.max()

    ymin = y.min()
    ymax = y.max()

    X, Y = numpy.mgrid[xmin:xmax:30j, ymin:ymax:30j] # type: ignore

    positions = numpy.vstack([X.ravel(), Y.ravel()])
    values = numpy.vstack([x, y])

    kernel = scipy.stats.gaussian_kde(values)
    Z = numpy.reshape(kernel(positions).T, X.shape)

    # assert False, dict(Z=Z.shape, positions=positions.shape)

    # z is density over the grid ((xmin, xmax) (ymin, ymax))
    if ravel:
        return X.ravel(), Y.ravel(), Z.ravel()
    return X, Y, Z

def gaussian_kde_2d_df(xys, key = "key", **kde_kwargs):
    ks = []
    xs = []
    ys = []
    densities = []
    for k, (x, y) in xys.items():
        _xs, _ys, ds = gaussian_kde_2d(x, y, **kde_kwargs)
        xs.extend(_xs)
        ys.extend(_ys)
        densities.extend(ds)
        ks.extend([k for _ in ds])
    return pandas.DataFrame({
        key: ks,
        "x": xs,
        "y": ys,
        "density": densities,
    })