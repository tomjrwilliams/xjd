
import datetime

import numpy
import pandas
import jax
import optax

import xtuples as xt
import xjd


def test_ppca() -> bool:
    xjd.utils.rand.reset_keys()

    N = 3

    ds = xjd.utils.dates.starting(datetime.date(2020, 1, 1), 100)

    vs_norm = xjd.utils.rand.gaussian((100, N,))

    N_COLS = 5
    betas = xjd.utils.rand.gaussian((N, N_COLS,))
    vs = numpy.matmul(vs_norm, betas)

    data = (
        pandas.DataFrame({
            f: xjd.utils.dates.dated_series({
                d: v for d, v in zip(ds, fvs) #
            })
            for f, fvs in enumerate(numpy.array(vs).T)
        }),
    )
    NOISE = 0

    model = xjd.Model()

    model, loc_data = model.add_node(
        xjd.inputs.dfs.DataFrame_Wide(),
        input=True,
    )
    model, loc_cov = model.add_node(
        xjd.cov.vanilla.Cov(data=loc_data.result()), static=True
    )
    model, loc_weights = model.add_node(
        xjd.params.random.Orthogonal(
            shape=(N_COLS, N + NOISE,)
        )
    )
    model, loc_eigval = model.add_node(
        xjd.params.random.Gaussian(shape=(N + NOISE,))
    )
    model, loc_eigval_sq = model.add_node(
        xjd.transforms.scaling.Sq(
            data=loc_eigval.param()
        ),
    )
    model, loc_encode = model.add_node(
        xjd.pca.vanilla.PCA_Encoder(
            data=loc_data.result(),
            weights=loc_weights.param(),
            n=N + NOISE,
            #
        )
    )
    model, loc_decode = model.add_node(
        xjd.pca.vanilla.PCA_Decoder(
            weights=loc_weights.param(),
            factors=loc_encode.result(),
            #
        )
    )
    model = model.add_node(
        xjd.constraints.linalg.Eigenvec(
            cov=loc_cov.result(),
            weights=loc_weights.param(),
            eigvals=loc_eigval_sq.result(),
        ),
        constraint=True,
    ).init(data)

    model = model.optimise(
        data, 
        iters = 2500,
        # max_error_unchanged=1000,
        rand_init=100,
        # opt = optax.sgd(.1),
        # opt=optax.noisy_sgd(.1),
        # jit=False,
    ).apply(data)

    eigen_vec = loc_weights.param().access(model)
    sigma = loc_eigval_sq.result().access(model)
    factors = loc_encode.result().access(model)

    # cov = jax.numpy.cov(factors.T)
    # eigen_vals = jax.numpy.diag(cov)

    print(numpy.round(
        numpy.matmul(eigen_vec.T, eigen_vec),
        2
    ))

    eigen_vals = sigma

    order = numpy.flip(numpy.argsort(eigen_vals))
    # assert eigen_vals.shape[0] == N, eigen_vals.shape

    eigen_vals = eigen_vals[order]
    eigen_vec = eigen_vec[..., order]

    eigvals, eigvecs = numpy.linalg.eig(numpy.cov(
        numpy.transpose(data[0].values)
    ))
    _order = numpy.flip(numpy.argsort(eigvals))
    
    eigvecs = eigvecs[..., _order]
    eigvals = eigvals[_order]

    eigvecs = xjd.utils.funcs.set_signs_to(
        eigvecs, 1, numpy.ones(eigvecs.shape[1])
    )
    eigen_vec = xjd.utils.funcs.set_signs_to(
        eigen_vec, 1, numpy.ones(eigen_vec.shape[1])
    )

    print(numpy.round(eigen_vec, 4))
    print(numpy.round(eigvecs, 4))

    print(numpy.round(eigen_vals, 3))
    print(numpy.round(eigvals, 3))

    xjd.utils.tests.assert_is_close(
        eigen_vec.real[..., :1],
        eigvecs.real[..., :1],
        True,
        atol=.1,
    )
    return True