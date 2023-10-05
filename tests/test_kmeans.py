
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

def test_kmeans() -> bool:
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
    model, loc_var = model.add_node(
        xf.params.random.Gaussian(
            shape=(N_CLUSTERS, N_COLS,),
        )
    )
    model, loc_label = model.add_node(
        xf.clustering.kmeans.KMeans_Labels(
            k=3,
            mu=loc_mu.param(),
            var=loc_var.param(),
            data=loc_data.result(),
        )
    )
    model, loc_EM = model.add_node(
        xf.clustering.kmeans.KMeans_EM_Naive(
            k=3,
            data=loc_data.result(),
            labels=loc_label.result(),
        )
    )
    model = (
        model.add_node(xf.constraints.em.EM(
            param=loc_mu.param(),
            optimal=loc_EM.result(0),
            cut_tree=True,
        ), constraint=True)
        .add_node(xf.constraints.em.EM(
            param=loc_var.param(),
            optimal=loc_EM.result(1),
            cut_tree=True,
        ), constraint=True)
        .init(data)
    )

    model = model.optimise(
        data,
        iters = 1000,
        opt=optax.noisy_sgd(.1),
        rand_init=100,
        # jit = False,
    ).apply(data)

    # mu
    clusters = loc_mu.param().access(model)
    labels = loc_label.result().access(model)
    
    labels, order = (
        xt.iTuple([int(l) for l in labels])
        .pipe(xf.clustering.kmeans.reindex_labels)
    )
    clusters = [clusters[i] for i in order]

    k_means = KMeans(n_clusters=3, random_state=69).fit(vs)
    sk_labels, sk_order = xt.iTuple(k_means.labels_).pipe(
        xf.clustering.kmeans.reindex_labels
    )

    clusters = numpy.round(clusters, 3)
    mu = numpy.round(mu, 3)
    
    assert labels == sk_labels, {
        i: (l, sk_l,) for i, (l, sk_l)
        in enumerate(zip(labels, sk_labels))
        if l != sk_l
    }

    xf.utils.tests.assert_is_close(
        clusters,
        mu,
        True,
        atol=0.2,
    )

    return True