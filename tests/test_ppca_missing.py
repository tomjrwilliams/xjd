


import datetime

import numpy
import pandas
import jax
import optax

import xtuples as xt
import xfactors as xf


def test_ppca_missing() -> bool:
    xf.utils.rand.reset_keys()

    N = 3

    ds = xf.utils.dates.starting(datetime.date(2020, 1, 1), 100)

    vs_norm = xf.utils.rand.gaussian((100, N,))

    N_COLS = 5
    betas = xf.utils.rand.gaussian((N, N_COLS,))
    vs = numpy.matmul(vs_norm, betas)

    exists = xf.utils.rand.bernoulli(p=0.9, shape=vs.shape)
    missing = 1 + (-1 * exists)

    vs_mask = numpy.where(missing, numpy.zeros(vs.shape), vs)

    data = (
        pandas.DataFrame({
            f: xf.utils.dates.dated_series({
                d: v for d, v in zip(ds, fvs) #
            })
            for f, fvs in enumerate(numpy.array(vs_mask).T)
        }),
        pandas.DataFrame({
            f: xf.utils.dates.dated_series({
                d: v for d, v in zip(ds, fvs) #
            })
            for f, fvs in enumerate(numpy.array(exists).T)
        }),
    )

    NOISE = 0

    model = xf.Model()

    model, loc_data_raw = model.add_node(
        xf.inputs.dfs.DataFrame_Wide(),
        input=True,
    )
    model, loc_exists = model.add_node(
        xf.inputs.dfs.DataFrame_Wide(),
        input=True,
    )
    model, loc_cov = model.add_node(
        xf.cov.vanilla.Cov(
            data=loc_data_raw.result(),
            exists=loc_exists.result(),
        ),
        static=True
    )
    model, loc_missing = model.add_node(
        xf.params.random.Gaussian(shape=vs.shape)
    )
    model, loc_data = model.add_node(
        xf.transforms.masks.Where(
            loc_exists.result(),
            loc_data_raw.result(),
            loc_missing.param(),
        )
    )
    model, loc_weights = model.add_node(
        xf.params.random.Orthogonal(
            shape=(N_COLS, N + NOISE,)
        )
    )
    model, loc_eigval = model.add_node(
        xf.params.random.Gaussian(shape=(N + NOISE,))
    )
    model, loc_eigval_sq = model.add_node(
        xf.transforms.scaling.Sq(
            data=loc_eigval.param()
        ),
    )
    model, loc_encode = model.add_node(
        xf.pca.vanilla.PCA_Encoder(
            data=loc_data.result(),
            weights=loc_weights.param(),
            n=N + NOISE,
            #
        )
    )
    model, loc_decode = model.add_node(
        xf.pca.vanilla.PCA_Decoder(
            weights=loc_weights.param(),
            factors=loc_encode.result(),
            #
        )
    )
    model = model.add_node(
        # NOTE: eigen vec seems to be too aggressive?
        # or depends too much on noisy covariance estimate?
        xf.constraints.linalg.Orthonormal(loc_weights.param(), T=True),
        # xf.constraints.linalg.Eigenvec(
        #     cov=loc_cov.result(),
        #     weights=loc_weights.param(),
        #     eigvals=loc_eigval_sq.result(),
        # ),
        constraint=True,
    ).add_node(
        xf.constraints.loss.MSE(
            loc_data.result(),
            loc_decode.result(),
            condition=loc_exists.result(),
        ),
        constraint=True,
    ).init(data)

    model = model.optimise(
        data, 
        iters = 2500,
        # max_error_unchanged=1000,
        rand_init=100,
        # opt = optax.sgd(.01),
        # opt=optax.noisy_sgd(.01),
        # jit=False,
    ).apply(data)

    eigen_vec = loc_weights.param().access(model)
    sigma = loc_eigval_sq.result().access(model)
    factors = loc_encode.result().access(model)

    print("missing:", missing.sum())

    missing_data = list(loc_missing.param().access(model).flatten())

    vs = list(vs.flatten())
    missing = list(missing.flatten())

    vs_missing, _, vs_res = (
        xt.iTuple(vs)
        .zip(missing, missing_data)
        .filterstar(
            lambda v, is_missing, res: int(is_missing)
        )
        .zip()
    )

    xf.utils.tests.assert_is_close(
        loc_data.result().access(model),
        loc_decode.result().access(model),
        atol=.2,
    )
    xf.utils.tests.assert_is_close(
        numpy.array(vs_missing),
        numpy.array(vs_res),
        atol=.1,
    )
    return True