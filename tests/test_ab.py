
import datetime

import numpy
import pandas
import jax
import optax

import xtuples as xt
import xfactors as xf


def test_ab() -> bool:
    xf.utils.rand.reset_keys()

    N = 100

    ds = xf.utils.dates.starting(datetime.date(2020, 1, 1), N)

    process = xf.utils.rand.gaussian((N,))
    v_series = xf.utils.dates.dated_series({
        d: v for d, v in zip(ds, process) #
    }).rolling("5D").mean().fillna(0)

    noise = xf.utils.rand.gaussian((N,)) / 3

    data = (
        pandas.DataFrame({
            0: pandas.Series(
                index=v_series.index,
                data=numpy.concatenate([
                    numpy.cumsum(v_series.values)
                ]) + noise,
            )
        }),
    )

    model = xf.Model()
    model, loc_data = model.add_node(
        xf.inputs.dfs.DataFrame_Wide(),
        input=True,
    )

    model, loc_position = model.add_node(
        xf.params.random.Gaussian((N+1, 1,))
    )
    model, loc_velocity = model.add_node(
        xf.params.random.Gaussian((N, 1,))
    )
    model, loc_alpha_raw = model.add_node(
        xf.params.random.Gaussian((1,))
    )
    model, loc_beta_raw = model.add_node(
        xf.params.random.Gaussian((1,))
    )

    model, loc_alpha = model.add_node(
        xf.transforms.scaling.Expit(loc_alpha_raw.param()),
    )
    model, loc_beta = model.add_node(
        xf.transforms.scaling.Expit(loc_beta_raw.param()),
    )

    model, loc_pred = model.add_node(
        xf.forecasting.ab.Prediction(
            position=loc_position.param(),
            velocity=loc_velocity.param(),
        )
    )
    model, loc_update = model.add_node(
        xf.forecasting.ab.Update(
            position=loc_data.result(),
            prediction=loc_pred.result(),
            velocity=loc_velocity.param(),
            alpha=loc_alpha.result(),
            beta=loc_beta.result(),
        ),
        markov=xt.iTuple((
            loc_position,
            loc_velocity,
        )),
    )
    model = (
        # model.add_node(
        #     xf.constraints.loss.MinimiseSquare(
        #         loc_update.result(2),
        #     ),
        #     constraint=True,
        # )
        model.add_node(
            xf.constraints.loss.MSE(
                loc_velocity.param(),
                loc_update.result(1),
            ),
            constraint=True,
        ).add_node(
            xf.constraints.loss.MSE(
                loc_pred.result(),
                loc_update.result(0),
            ),
            constraint=True,
        )
    ).init(data)

    model = model.optimise(
        data, 
        iters = 2500,
        max_error_unchanged=0.5,
        rand_init=100,
        # opt = optax.sgd(.1),
        # opt=optax.noisy_sgd(.1),
        jit=False,
    ).apply(data)

    alpha = loc_alpha.result().access(model)
    beta = loc_beta.result().access(model)

    position = loc_position.param().access(model)

    xf.utils.tests.assert_is_close(
        numpy.round(data[0][0].values, 2),
        numpy.round(position[:, 0], 2)[:-1],
        atol=.1,
    )

    return True