
import datetime

import numpy
import pandas
import jax
import optax

import xtuples as xt
import xfactors as xf


# jax.config.update("jax_debug_nans", True)

def test_kf() -> bool:
    xf.utils.rand.reset_keys()

    N = 100

    ds = xf.utils.dates.starting(datetime.date(2020, 1, 1), N)

    process = numpy.array(xf.utils.rand.gaussian((N,)))
    v_series = xf.utils.dates.dated_series({
        d: v for d, v in zip(ds, process) #
    })
    # .rolling("5D").mean().fillna(0)

    noise = xf.utils.rand.gaussian((N,)) / 3

    factors = (
        pandas.DataFrame({
            0: v_series,
            1: pandas.Series(
                index=v_series.index,
                data=numpy.concatenate([
                    numpy.zeros(1),
                    numpy.cumsum(v_series.values)[:-1],
                ]) + noise,
            )
        }),
    )

    N_COLS = 3
    # betas = xf.utils.rand.gaussian((1, N_COLS,))
    betas = xf.utils.rand.gaussian((2, N_COLS,))
    # betas = numpy.vstack([
    #     numpy.zeros(betas.shape),
    #     betas,
    # ])

    vs = numpy.matmul(factors[0].values, betas)
    v_noise = xf.utils.rand.gaussian(vs.shape) / 3

    vs = vs + v_noise

    data = (
        pandas.DataFrame({
            f: xf.utils.dates.dated_series({
                d: v for d, v in zip(ds, fvs) #
            })
            for f, fvs in enumerate(numpy.array(vs).T)
        }),
    )

    model = xf.Model()
    model, loc_data = model.add_node(
        xf.inputs.dfs.DataFrame_Wide(),
        input=True,
    )

    model, loc_noise = model.add_node(
        xf.params.random.Gaussian((N + 1, 2,))
    )
    model, loc_state = model.add_node(
        xf.params.random.Gaussian((N + 1, 2,))
    )
    # model, loc_cov = model.add_node(
    #     xf.params.scalar.Scalar(
    #         xf.expand_dims(jax.numpy.eye(2), 0, N+1)
    #     )
    # )
    model, loc_transition_raw = model.add_node(
        xf.params.random.Gaussian((2, 2,))
    )
    model, loc_transition_mask = model.add_node(
        xf.transforms.masks.Zero(
            loc_transition_raw.param(),
            numpy.array([
                [0., 0.],
                [1., 1.]
            ])
        )
    )
    model, loc_transition = model.add_node(
        xf.transforms.masks.Positive(
            loc_transition_mask.result(),
            numpy.array([
                [0., 1.],
                [1., 1.]
            ])
        )
    )
    model, loc_observation_raw = model.add_node(
        xf.params.random.Gaussian((N_COLS, 2,))
    )
    model, loc_observation_pos = model.add_node(
        xf.transforms.masks.Positive(
            loc_observation_raw.param(),
            numpy.array([
                [0, 0, 0],
                [1, 1, 1],
            ]).T
        )
    )
    model, loc_observation = model.add_node(
        xf.transforms.masks.Negative(
            loc_observation_pos.result(),
            numpy.array([
                [1, 1, 1],
                [0, 0, 0],
            ]).T
        )
    )
    
    model, loc_state_eigvec_raw = model.add_node(
        xf.params.scalar.Scalar(
            xf.expand_dims(numpy.eye(2), 0, N+1)
        )
    )
    model, loc_noise_state_eigvec_raw = model.add_node(
        xf.params.scalar.Scalar(numpy.eye(2))
    )
    model, loc_noise_obs_eigvec_raw = model.add_node(
        xf.params.scalar.Scalar(numpy.eye(3))
    )

    model, loc_state_eigvec = model.add_node(
        xf.transforms.scaling.UnitNorm(
            loc_state_eigvec_raw.param(),
        )
    )
    model, loc_noise_state_eigvec = model.add_node(
        xf.transforms.scaling.UnitNorm(
            loc_noise_state_eigvec_raw.param(),
        )
    )
    model, loc_noise_obs_eigvec = model.add_node(
        xf.transforms.scaling.UnitNorm(
            loc_noise_obs_eigvec_raw.param(),
        )
    )


    model, loc_state_eigval_raw = model.add_node(
        xf.params.random.Gaussian((N+1, 2,))
    )
    model, loc_noise_state_eigval_raw = model.add_node(
        xf.params.random.Gaussian((2,))
    )
    model, loc_noise_obs_eigval_raw = model.add_node(
        xf.params.random.Gaussian((3,))
    )
    model, loc_state_eigval = model.add_node(
        xf.transforms.scaling.Sq(
            loc_state_eigval_raw.param(),
            vmin=0.01
        )
    )
    model, loc_noise_state_eigval = model.add_node(
        xf.transforms.scaling.Sq(
            loc_noise_state_eigval_raw.param(),
            vmin=0.01
        )
    )
    model, loc_noise_obs_eigval = model.add_node(
        xf.transforms.scaling.Sq(
            loc_noise_obs_eigval_raw.param(),
            vmin=0.01
        )
    )

    model, loc_cov = model.add_node(
        xf.transforms.linalg.Eigen_Cov(
            loc_state_eigval.result(),
            loc_state_eigvec.result(),
            # vmax=3.,
        )
    )
    model, loc_noise_state = model.add_node(
        xf.transforms.linalg.Eigen_Cov(
            loc_noise_state_eigval.result(),
            loc_noise_state_eigvec.result(),
            # vmax=3.,
        )
    )
    model, loc_noise_obs = model.add_node(
        xf.transforms.linalg.Eigen_Cov(
            loc_noise_obs_eigval.result(),
            loc_noise_obs_eigvec.result(),
            # vmax=3.,
        )
    )

    model, loc_state_pred = model.add_node(
        xf.forecasting.kf.State_Prediction(
            transition=loc_transition.result(),
            state=loc_state.param(),
            noise=loc_noise.param(),
            # control = loc_control.param(),
            # input=loc_input.param(),
        ),
    )
    model, loc_cov_pred = model.add_node(
        xf.forecasting.kf.Cov_Prediction(
            transition=loc_transition.result(),
            cov=loc_cov.result(),
            noise=loc_noise_state.result()
        )
    )
    model, loc_state_inn = model.add_node(
        xf.forecasting.kf.State_Innovation(
            data=loc_data.result(),
            observation=loc_observation.result(),
            state=loc_state_pred.result(1),
        )
    )
    model, loc_cov_inn = model.add_node(
        xf.forecasting.kf.Cov_Innovation(
            observation=loc_observation.result(),
            cov=loc_cov_pred.result(0),
            noise = loc_noise_obs.result(),
        ),
    )
    model, loc_kg = model.add_node(
        xf.forecasting.kf.Kalman_Gain(
            observation=loc_observation.result(),
            cov=loc_cov_pred.result(0),
            cov_innovation=loc_cov_inn.result(),
        ),
    )
    model, loc_state_new = model.add_node(
        xf.forecasting.kf.State_Updated(
            state=loc_state_pred.result(1),
            kalman_gain=loc_kg.result(),
            state_innovation=loc_state_inn.result(),
        ),
        # markov=loc_state.param(),
    )
    model, loc_cov_new = model.add_node(
        xf.forecasting.kf.Cov_Updated(
            kalman_gain=loc_kg.result(),
            observation=loc_observation.result(),
            cov=loc_cov_pred.result(0),
        ),
        # markov=loc_cov.result(), 
    )

    model, loc_residual = model.add_node(
        xf.forecasting.kf.Residual(
            data=loc_data.result(),
            observation=loc_observation.result(),
            state=loc_state_new.result(),
        )
    )

    # TODO: parametrise in the noise

    model = (
        model.add_node(
            xf.constraints.loss.L2(
                loc_state_inn.result(),
            ),
            constraint=True,
        )
        .add_node(
            xf.constraints.loss.MSE(
                loc_cov_pred.result(1),
                loc_noise_state.result(),
                # mse(next - pred, noise)
            ),
            constraint=True,
        )
        .add_node(
            xf.constraints.loss.L2(
                loc_state_pred.result(2),
            ),
            constraint=True,
        )
        .add_node(
            xf.constraints.loss.L2(
                loc_noise_obs_eigval.result(),
            ),
            constraint=True,
        )
        .add_node(
            xf.constraints.linalg.Orthonormal(
                loc_state_eigvec.result(),
            ),
            constraint=True,
        )
        .add_node(
            xf.constraints.linalg.Orthonormal(
                loc_noise_state_eigvec.result(),
            ),
            constraint=True,
        )
        .add_node(
            xf.constraints.linalg.Orthonormal(
                loc_noise_obs_eigvec.result(),
            ),
            constraint=True,
        )
    ).init(data)

    # NOTE: under determined up to scaling in the example above
    # and given the linearity, probably many cases

    # so perhaps have some kind of parameter regularisation
    # could be unit norm, or perhaps just gaussian likelihood
    # independent, unit scale variance

    # (which mimics how the test data was generated)

    model = model.optimise(
        data, 
        iters = 2500,
        max_error_unchanged=0.5,
        rand_init=100,
        # opt = optax.sgd(.1),
        opt=optax.sgd(.1),
        # jit=False,
    ).apply(data)

    transition = numpy.round(
        loc_transition.result().access(model), 2
    )
    observation = numpy.round(
        loc_observation.result().access(model), 2
    ).T

    noise_obs = numpy.round(
        loc_noise_obs.result().access(model), 2
    )
    noise_state = numpy.round(
        loc_noise_state.result().access(model), 2
    )

    betas = numpy.round(betas, 2)

    factor_var = numpy.var(factors[0].values, axis=0)
    v_var = numpy.var(v_noise, axis=0)

    noise = loc_noise.param().access(model)[:-1, :]
    state = loc_state.param().access(model)[:-1, :]
    state_pred = loc_state_pred.result(1).access(model)[:-1, :]

    deltas = jax.numpy.concatenate([
        # noise,
        factors[0].values,
        state,
        data[0].values,
        numpy.matmul(
            observation.T, xf.expand_dims(state, -1, 1)
        )[:, :, 0],
        numpy.matmul(
            observation.T, xf.expand_dims(state_pred, -1, 1)
        )[:, :, 0]
    ], axis=1)

    kg = loc_kg.result().access(model)
    print(numpy.round(kg.mean(axis=0), 3))

    max_digits = len("%.2f" % numpy.max(numpy.abs(deltas)))

    f_round = lambda v, n, bounds: " ".join([
        xf.visuals.formatting.left_pad(
            ('%' + ".{}f".format(n)) % round(_v, n),
            max_digits,
            "0"
        ) + (
            " | " if i in bounds else ""
        )
        for i, _v in enumerate(v)
    ])

    for r in deltas:
        print(f_round(r, 2, [1, 3, 3 + 3, 3 + 6]))

    for k, v in {
        "transition": transition,
        "betas": betas,
        "observation": observation,
        "noise_state": noise_state,
        "noise_obs": noise_obs,
        "factor_var": factor_var,
        "v_var": v_var,
        "noise_state_eigval": numpy.round(
            loc_noise_state_eigval.result().access(model), 2
        ),
        "noise_obs_eigval": numpy.round(
            loc_noise_obs_eigval.result().access(model), 2
        ),
    }.items(): print(k, v)

    xf.utils.tests.assert_is_close(betas, observation, atol=.1)
    xf.utils.tests.assert_is_close(
        xf.utils.funcs.to_diag(v_var), noise_obs, atol=.1
    )

    assert False

    # target transition model =
    # new_0 = [0 * old_0, 0 * old_1]
    # new_1 = [1 * old_1, 1 * old_0]

    # target observation model = betas

    # loc_noise_state = 
    # diagonal
    # 0: variance = 1
    # 1: varaince = 1 / 3 (?)

    # loc_noise_obs = 
    # diagonal
    # variance = 1 / 3

    return True 