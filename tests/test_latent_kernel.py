
import itertools
import datetime

import numpy
import pandas

import xtuples as xt
import xfactors as xf


# import jax.config
# jax.config.update("jax_debug_nans", True)

from sklearn.cluster import KMeans

def test_latent_kernel() -> bool:
    xf.utils.rand.reset_keys()

    ds = xf.utils.dates.starting(datetime.date(2020, 1, 1), 100)

    N_COLS = 5
    N_CLUSTERS = 3
    N_FACTORS = 5
    N_VARIABLES = N_CLUSTERS * N_COLS

    VARIABLES = xt.iTuple.range(N_VARIABLES)
    CLUSTERS = xt.iTuple.range(N_CLUSTERS)
    FACTORS  = xt.iTuple.range(N_FACTORS)

    CLUSTER_MEMBERS = {
        i: chunk
        for i, chunk 
        in VARIABLES.chunkby(lambda i: i // N_COLS).enumerate()
    }
    CLUSTER_MAP = xt.iTuple(CLUSTER_MEMBERS.items()).fold(
        lambda acc, cluster_chunk: {
            **acc,
            **{
                i: cluster_chunk[0]
                for i in cluster_chunk[1]
            }
        },
        initial={},
    )

    # mu = xf.utils.rand.orthogonal(N_FACTORS)[..., :N_CLUSTERS]
    mu = xf.utils.rand.gaussian((N_FACTORS, N_CLUSTERS,))
    
    betas = numpy.add(
        numpy.array([
            [
                mu[f][CLUSTER_MAP[i]]
                for i in VARIABLES
            ]
            for f in FACTORS
        ]),
        xf.utils.rand.gaussian((N_FACTORS, N_VARIABLES,)) / 10
    ).T
    cov = numpy.matmul(
        numpy.matmul(betas, numpy.eye(N_FACTORS)), betas.T
    )
    cov = numpy.divide(
        cov, 
        xf.expand_dims_like(
            cov.sum(axis=1), axis=1, like=cov
        ),
        # numpy.resize(
        #     numpy.expand_dims(cov.sum(axis=1), axis=1),
        #     cov.shape,
        # )
    )

    vs = xf.utils.rand.v_mv_gaussian(
        100,
        mu=numpy.zeros((N_VARIABLES,)), 
        cov=cov
    )
    assert not numpy.isnan(vs).any(), betas

    data = (
        pandas.DataFrame({
            f: xf.utils.dates.dated_series({
                d: v for d, v in zip(ds, fvs)
                #
            })
            for f, fvs in enumerate(numpy.array(vs).T)
        }),
    )
    
    model = xf.Model()

    model, loc_data = model.add_node(
        xf.inputs.dfs.DataFrame_Wide(),
        input=True,
    )
    model, loc_cov = model.add_node(
        xf.cov.vanilla.Cov(data=loc_data.result()),
        static=True,
    )
    model, loc_latent = model.add_node(
        xf.params.latent.Latent(
            n=2,
            axis=1,
            data=loc_data.result(),
        )
    )
    model, loc_kernel = model.add_node(
        xf.reg.gp.GP_RBF(
            # sigma=1.,
            features=loc_latent.param(),
        )
    )
    model = model.add_node(
        xf.constraints.loss.MSE(
            l=loc_cov.result(),
            r=loc_kernel.result(),
        ),
        constraint=True,
    ).init(data)

    model = model.optimise(data, iters = 2500).apply(data)

    latents = loc_latent.param().access(model)
    cov_res = loc_kernel.result().access(model)

    k_means = KMeans(n_clusters=3, random_state=69).fit(latents)
    
    labels = xt.iTuple(k_means.labels_)
    labels_ordered = {
        label: i for i, label in enumerate(sorted(
            set(labels),
            key=labels.index
        ))
    }
    labels = labels.map(lambda l: labels_ordered[l])
    label_map = {i: l for i, l in labels.enumerate()}

    assert label_map == CLUSTER_MAP, dict(
        labels=label_map,
        clusters=CLUSTER_MAP,
    )

    xf.utils.tests.assert_is_close(
        cov,
        cov_res,
        True,
        atol=.05,
        n_max=10,
    )

    return True