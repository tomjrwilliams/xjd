
import datetime

import numpy
import pandas

import xtuples as xt
import xfactors as xf


def test_linreg() -> bool:
    xf.utils.rand.reset_keys()

    ds = xf.utils.dates.starting(datetime.date(2020, 1, 1), 100)

    vs_i = xf.utils.rand.gaussian((100, 3,))
    betas = xf.utils.rand.gaussian((3, 1,))
    vs_o = numpy.matmul(vs_i, betas)

    data = (
        pandas.DataFrame({
            f: xf.utils.dates.dated_series({
                d: v for d, v in zip(ds, fvs)
                #
            })
            for f, fvs in enumerate(numpy.array(vs_i).T)
        }),
        pandas.DataFrame({
            f: xf.utils.dates.dated_series({
                d: v for d, v in zip(ds, fvs)
                #
            })
            for f, fvs in enumerate(numpy.array(vs_o).T)
        }),
    )

    model, loc_input = xf.Model().add_node(
        xf.inputs.dfs.DataFrame_Wide(),
        input=True,
    )
    model, loc_output = model.add_node(
        xf.inputs.dfs.DataFrame_Wide(),
        input=True,
    )
    model, loc_weights = model.add_node(
        xf.params.random.Gaussian(shape=(3, 1,),)
    )
    model, loc_reg = model.add_node(
        xf.reg.lin.Lin_Reg(
            n=1, 
            data=loc_input.result(),
            weights=loc_weights.param(),
        ),
    )
    model = (
        model.add_node(xf.constraints.loss.MSE(
            l=loc_output.result(),
            r=loc_reg.result()
        ), constraint=True)
        .init(data)
    )

    betas_pre = loc_weights.param().access(model)
    model = model.optimise(data).apply(data)
    betas_post = loc_weights.param().access(model)

    results = dict(
        betas=betas.squeeze(),
        pre=betas_pre.squeeze(),
        post=betas_post.squeeze(),
    )

    xf.utils.tests.assert_is_close(
        results["betas"],
        results["pre"],
        False,
        results,
    )
    xf.utils.tests.assert_is_close(
        results["betas"],
        results["post"],
        True,
        results,
    )

    return True
    