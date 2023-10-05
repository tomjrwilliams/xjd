
import itertools
import datetime

import numpy
import pandas

import xtuples as xt
import xfactors as xf


import optax

# import jax.config
# jax.config.update("jax_debug_nans", True)

from sklearn.cluster import KMeans

def test_gmm() -> bool:
    xf.utils.rand.reset_keys()

    N_COLS = 5
    N_CLUSTERS = 3
    N_VARIABLES = 30

    mu = numpy.stack([
        numpy.ones(N_COLS) * -1,
        numpy.zeros(N_COLS),
        numpy.ones(N_COLS) * 1,
    ]) + (xf.utils.rand.gaussian((N_CLUSTERS, N_COLS,)) / 2)

    vs = numpy.concatenate([
        mu[cluster] + (xf.utils.rand.gaussian(
            (N_VARIABLES, N_COLS)
            #
        ) / 2)
        for cluster in range(N_CLUSTERS)
    ], axis = 0)

    data = (
        pandas.DataFrame({
            f: pandas.Series(
                index=list(range(len(fvs))),
                data=fvs,
                #
            )
            for f, fvs in enumerate(numpy.array(vs).T)
        }),
    )

    model = xf.Model()
    
    model, loc_data = model.add_node(
        xf.inputs.dfs.DataFrame_Wide(),
        input=True,
    )
    model, loc_mu = model.add_node(
        xf.params.random.Gaussian(
            shape=(N_CLUSTERS, N_COLS,),
        )
    )
    model, loc_cov = model.add_node(
        xf.params.random.Orthogonal(
            shape=(N_CLUSTERS, N_COLS, N_COLS,),
        )
    )
    model, loc_gmm = model.add_node(
        xf.clustering.gmm.Likelihood_Separability(
            k=N_CLUSTERS,
            data=loc_data.result(),
            mu=loc_mu.param(),
            cov=loc_cov.param(),
        ), 
        random = True
    )
    model = (
        model.add_node(
            xf.constraints.loss.Maximise(
                data=loc_gmm.result(1),
            ),
            constraint=True,
        )
        .add_node(
            xf.constraints.loss.Maximise(
                data=loc_gmm.result(2),
            ),
            constraint=True,
        )
        .add_node(
            xf.constraints.linalg.VOrthogonal(
                data=loc_cov.param()
            ),
            constraint=True,
        )
        .add_node(
            xf.constraints.linalg.L1_MM_Diag(
                raw=loc_cov.param()
            ),
            constraint=True,
        )
        .init(data)
    )

    # from jax.config import config 
    # config.update("jax_debug_nans", True) 

    model = model.optimise(
        data,
        iters = 1000,
        opt=optax.noisy_sgd(.1),
        max_error_unchanged = 0.5,
        rand_init=1000,
        # jit = False,
    ).apply(data)

    mu_ = loc_mu.param().access(model)
    cov_ = loc_cov.param().access(model)
    # probs = params[2]
    probs = loc_gmm.result(0).access(model)
    
    cov_ = numpy.round(numpy.matmul(
        numpy.transpose(cov_, (0, 2, 1)),
        cov_,
    ), 3)

    labels = probs.argmax(axis=1)
    # n_data

    # print(cov_)
    # print(labels)
    # print(mu_)
    # print(mu)
    
    # print(results[EM][0][3])
    # print(results[EM][0][0])
    
    labels, order = (
        xt.iTuple([int(l) for l in labels])
        .pipe(xf.clustering.kmeans.reindex_labels)
    )
    mu_ = [mu_[i] for i in order]

    k_means = KMeans(n_clusters=3, random_state=69).fit(vs)
    sk_labels, sk_order = xt.iTuple(k_means.labels_).pipe(
        xf.clustering.kmeans.reindex_labels
    )

    mu_ = numpy.round(mu_, 3)
    mu = numpy.round(mu, 3)
    
    assert labels == sk_labels, {
        i: (l, sk_l,) for i, (l, sk_l)
        in enumerate(zip(labels, sk_labels))
        if l != sk_l
    }

    xf.utils.tests.assert_is_close(
        mu_,
        mu,
        True,
        atol=0.2,
    )

    return True