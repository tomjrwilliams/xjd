
import functools
import datetime

import numpy
import pandas
import jax
import optax

import xtuples as xt
import xjd

import optax
import jaxopt


def order_eig(evals, evecs):
    order = numpy.flip(numpy.argsort(evals))
    evecs = evecs[..., order]
    evecs = xjd.utils.funcs.set_signs_to(
        evecs, 1, numpy.ones(evecs.shape[1])
    )
    evals = evals[order]
    return evals, evecs

def test_eig(iters=2500) -> bool:
    xjd.utils.rand.reset_keys()

    eigvals = jax.numpy.square(xjd.utils.rand.gaussian((5,)))

    f_loss= functools.partial(
        xjd.utils.funcs.loss_eigvec_diag,
        eigvals=eigvals
    )
    # f_loss = lambda w_e: xjd.utils.funcs.loss_eigvec_diag(*w_e)

    w = xjd.utils.rand.gaussian((5, 5,))

    opt = optax.adam(0.01)
    solver = jaxopt.OptaxSolver(
        opt=opt, fun=f_loss, maxiter=1000, jit=True
    )

    params = w
    # params = (w, eigvals)
    
    state = solver.init_state(params) 

    for iter in range(iters):
        params, state = solver.update(
            params,
            state,
        )
        error = state.error
        if iter % int(iters / 10) == 0:
            print(iter, ":", error)

    w = params
    # w, eigvals = params

    eigvals, w = order_eig(eigvals, w)

    wTw = numpy.matmul(w.T, w)

    evals, evecs = numpy.linalg.eig(
        numpy.matmul(numpy.matmul(
            w, eigvals * numpy.eye(w.shape[0])
        ), w.T)
    )
    evals, evecs = order_eig(evals, evecs)

    print(numpy.round(wTw, 2))

    print(numpy.round(w, 2))
    print(numpy.round(evecs, 2))

    print(numpy.round(eigvals, 2))
    print(numpy.round(evals, 2))

    xjd.utils.tests.assert_is_close(
        eigvals,
        evals,
        True,
        atol=0.2,
    )

    xjd.utils.tests.assert_is_close(
        w,
        evecs,
        True,
        atol=0.2,
    )

    return True