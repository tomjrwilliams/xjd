
import datetime

import numpy
import pandas
import jax

import xtuples as xt
import xjd


def test_ppca_naive() -> bool:
    xjd.utils.rand.reset_keys()

    N = 3

    ds = xjd.utils.dates.starting(datetime.date(2020, 1, 1), 100)

    N_COLS = 5

    vs_norm = xjd.utils.rand.gaussian((100, N,))
    betas = xjd.utils.rand.gaussian((N, N_COLS,))
    vs = numpy.matmul(vs_norm, betas)

    NOISE = 1

    data = (
        pandas.DataFrame({
            f: xjd.utils.dates.dated_series({d: v for d, v in zip(ds, fvs)})
            for f, fvs in enumerate(numpy.array(vs).T)
        }),
    )

    model, loc_data = xjd.Model().add_node(
        xjd.inputs.dfs.DataFrame_Wide(),
        input=True,
    )
    model, loc_weights = model.add_node(
        xjd.params.random.Orthogonal(
            shape=(N_COLS, N + NOISE,)
        )
    )
    model, loc_encode = model.add_node(
        xjd.pca.vanilla.PCA_Encoder(
            data=loc_data.result(),
            weights = loc_weights.param(),
            n=N + NOISE,
            #
        )
    )
    model, loc_decode = model.add_node(
        xjd.pca.vanilla.PCA_Decoder(
            weights = loc_weights.param(),
            factors=loc_encode.result(),
            #
        )
    )
    model = (
        model.add_node(xjd.constraints.loss.MSE(
            l=loc_data.result(),
            r=loc_decode.result(),
        ), constraint=True)
        .add_node(xjd.constraints.linalg.EigenVLike(
            weights = loc_weights.param(),
            factors=loc_encode.result(),
            n_check=N + NOISE,
        ), constraint=True)
        .init(data)
    )

    model = model.optimise(data).apply(data)
    
    eigen_vec = weights = loc_weights.param().access(model)
    factors = loc_encode.result().access(model)

    cov = jax.numpy.cov(factors.T)
    eigen_vals = jax.numpy.diag(cov)

    order = numpy.flip(numpy.argsort(eigen_vals))[:N]
    assert eigen_vals.shape[0] == N + 1, eigen_vals.shape

    eigen_vals = eigen_vals[order]
    eigen_vec = eigen_vec[..., order]

    eigvals, eigvecs = numpy.linalg.eig(numpy.cov(
        numpy.transpose(data[0].values)
    ))
    _order = numpy.flip(numpy.argsort(eigvals))[:N]
    eigvecs = eigvecs[..., _order]
    # assert False, (eigvals, eigen_vals,)

    eigvecs = xjd.utils.funcs.set_signs_to(
        eigvecs, 1, numpy.ones(eigvecs.shape[1])
    )
    eigen_vec = xjd.utils.funcs.set_signs_to(
        eigen_vec, 1, numpy.ones(eigen_vec.shape[1])
    )

    print(eigen_vec)
    print(eigvecs)

    # for now we just check pc1 matches
    xjd.utils.tests.assert_is_close(
        eigen_vec.real[..., :1],
        eigvecs.real[..., :1],
        True,
        atol=.1,
    )
    return True