
import datetime

import numpy
import pandas

import xtuples as xt
import xjd


def test_pca() -> bool:
    xjd.utils.rand.reset_keys()

    ds = xjd.utils.dates.starting(datetime.date(2020, 1, 1), 100)

    vs_norm = xjd.utils.rand.gaussian((100, 3,))
    betas = xjd.utils.rand.gaussian((3, 5,))
    vs = numpy.matmul(vs_norm, betas)

    data = (
        pandas.DataFrame({
            f: xjd.utils.dates.dated_series({
                d: v for d, v in zip(ds, fvs)
                #
            })
            for f, fvs in enumerate(numpy.array(vs).T)
        }),
    )

    model, loc_data = xjd.Model().add_node(
        xjd.inputs.dfs.DataFrame_Wide(),
        input=True,
    )
    model, loc_pca = model.add_node(
        xjd.pca.vanilla.PCA(n=3, data = loc_data.result())
    )
    model = model.init(data)

    model = model.optimise(data).apply(data)
    
    eigen_val = loc_pca.result(0).access(model)
    eigen_vec = loc_pca.result(1).access(model)

    eigvals, eigvecs = numpy.linalg.eig(numpy.cov(
        numpy.transpose(data[0].values)
    ))

    # multiply by root(eigenval) -> beta?

    xjd.utils.tests.assert_is_close(
        eigen_val[:3],
        eigvals.real[:3],
        True,
    )
    xjd.utils.tests.assert_is_close(
        eigen_vec.real[..., :3],
        eigvecs.real[..., :3],
        True,
    )

    return True