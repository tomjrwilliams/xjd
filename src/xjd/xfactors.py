from __future__ import annotations

import enum

import operator
import collections
# import collections.abc
import functools
import itertools
from re import A
from tkinter import Y

import typing
import datetime

import numpy
import pandas

import jax
import jax.numpy
import jax.numpy.linalg

import jaxopt
import optax

import xtuples as xt

from . import utils

# ---------------------------------------------------------------

unsqueeze = utils.shapes.unsqueeze
expand_dims = utils.shapes.expand_dims
expand_dims_like = utils.shapes.expand_dims_like

# ---------------------------------------------------------------

SITE = 0
PARAM = 1
RESULT = 2
RANDOM = 3

def check_location(loc: Location):
    assert loc.domain in [
        SITE,
        PARAM,
        RESULT,
        RANDOM,
    ], loc
    return True

# ---
    
@xt.nTuple.decorate()
class Model(typing.NamedTuple):

    sites: xt.iTuple = xt.iTuple()
    params: xt.iTuple = xt.iTuple()
    results: xt.iTuple = xt.iTuple()
    random: xt.iTuple = xt.iTuple()

    order: xt.iTuple = xt.iTuple()
    stages: xt.iTuple = xt.iTuple()

    def add_node(
        self: Model, 
        node: Node, 
        **kws,
        #
    ) -> Model_w_Location:
        return add_node(self, node, **kws)

    def init(self: Model, data: tuple) -> Model:
        return init_model(self, data)

    def apply_flags(self: Model, **flags) -> Model:
        return apply_flags(self, **flags)

    def init_objective(self, *args, **kwargs):
        return init_objective(self, *args, **kwargs)

    def optimise(self, *args, **kwargs) -> Model:
        return optimise_model(self, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return apply_model(self, *args, **kwargs)

    # TODO: render method
    # that renders the graph
    # can use type(node).__name__ for labelling

# ---
    
@xt.nTuple.decorate()
class Model_w_Location(typing.NamedTuple):

    # NOTE: so we can method chain the model building

    model: Model
    loc: Location

    def add_node(
        self: Model_w_Location,
        node: Node,
        **kws,
        #
    ) -> Model_w_Location:
        return self.model.add_node(node, **kws)

    def init(self: Model_w_Location, data: tuple) -> Model:
        return self.model.init(data)

# ---------------------------------------------------------------

@xt.nTuple.decorate()
class Location(typing.NamedTuple):

    domain: int
    i: int
    path: xt.iTuple = xt.iTuple()

    # ---
    
    def check(self, *args, **kwargs):
        return check_location(self, *args, **kwargs)
        
    def access(self, *args, **kwargs):
        return access(self, *args, **kwargs)

    # ---

    def site(self, *path):
        return Location(SITE, self.i, self.path.extend(path))

    def param(self, *path):
        return Location(PARAM, self.i, self.path.extend(path))

    def result(self, *path):
        return Location(RESULT, self.i, self.path.extend(path))

    def random(self, *path):
        return Location(RANDOM, self.i, self.path.extend(path))

    @classmethod
    def SITE(cls, i, *path):
        return cls(SITE, i, xt.iTuple(path))

    @classmethod
    def PARAM(cls, i, *path):
        return cls(PARAM, i, xt.iTuple(path))

    @classmethod
    def RESULT(cls, i, *path):
        return cls(RESULT, i, xt.iTuple(path))

    @classmethod
    def RANDOM(cls, i, *path):
        return cls(RANDOM, i, xt.iTuple(path))

# ---
    
Loc = Location
OptionalLocation = typing.Optional[Location]
OptionalLoc = typing.Optional[Location]

# ---
    
import inspect

def access(
    loc: OptionalLoc, 
    model: Model,
    into: typing.Optional[typing.Type[LocationValue]] = None,
) -> LocationValue:
    assert loc is not None
    domain, i, path = loc
    if domain == 0:
        # assert no sub indices?
        res = model[domain][i]
    else:
        i = model.order[i]
        try:
            initial = model[domain][i]
        except:
            assert False, [domain, i, len(model[domain])]
        if not len(path):
            res = initial
        else:
            res = path.fold(
                lambda acc, _i: acc[_i], initial=initial
            )
    if into is not None:
        if inspect.isclass(into):
            if issubclass(into, xt.iTuple):
                res = into(res)
            assert isinstance(res, into), type(res)
    return res

# ---------------------------------------------------------------

import abc

class Node(typing.Protocol):

    @abc.abstractmethod
    def init(
        self: NodeClass,
        site: Site,
        model: Model,
        data = None,
    ) -> tuple[NodeClass, tuple, SiteValue]:
        ...

    @abc.abstractmethod
    def apply(
        self: NodeClass,
        site: Site,
        state: Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        ...

# TODO: could do auto shape checking fairly easily? optionally presumably

NodeClass = typing.TypeVar("NodeClass", bound=Node)

def init_null(
    self, site: Site, model: Model, data: tuple
) -> tuple[Node, tuple, SiteValue]:
    return self, (), ()

# ---------------------------------------------------------------

@xt.nTuple.decorate()
class Site(typing.NamedTuple):
    
    node: Node

    loc: typing.Optional[Location] = None
    shape: typing.Optional[xt.iTuple] = None

    input: bool = False
    constraint: bool = False
    random: bool = False
    static: bool = False
    masked: bool = False
    markov: bool = False
    
    # deciding whether to run
    only_if: dict = {}
    not_if: dict = {}
    any_if: dict = {} # note: only applied after the first two

    # what to return if we don't run
    if_not: typing.Union[tuple, float, jax.numpy.ndarray] = ()

    def should_run(self, flags):
        if len(self.only_if) == len(self.not_if) == 0:
            return True
        for k, v in self.only_if.items():
            if flags.get(k, None) != v:
                return False
        for k, v in self.not_if.items():
            if flags.get(k, None) == v:
                return False
        for k, v in self.any_if.items():
            if flags.get(k, None) == v:
                return True
        return True

    def apply_flags(self, flags):
        return self._replace(
            masked=True
        ) if not self.should_run(flags) else self

    def init(self, model: Model, data=None):
        res = self.node.init(self, model, data=data)
        assert res is not None, self.node # need to implement init
        # TODO: proper error message
        node, shape, params = res
        return self._replace(
            node=node,
            shape=xt.iTuple(shape),
        ), params

    def apply(
        self,
        state: Model,
        data=None,
    ) -> typing.Union[SiteValue, float]:
        if self.masked:
            return self.if_not
        return self.node.apply(self, state, data=data)

    # state or Model?
    def access(
        self, 
        model: Model,
        into: typing.Optional[typing.Type[LocationValue]] = None
    ):
        assert self.loc is not None
        return self.loc.access(model, into=into)

# ---
    
OptionalSite = typing.Optional[Site]
SiteValue = typing.Union[tuple, xt.iTuple, jax.numpy.ndarray]
LocationValue = typing.Union[Site, SiteValue]

# ---
    
def is_loc_field(annotation):
    if isinstance(annotation, typing.ForwardRef):
        return "Loc" in str(annotation)
    if isinstance(annotation, type):
        cls = annotation
        return cls is Location or issubclass(cls, Location)
    return Location in annotation.__args__

def loc_fields(node: Node):
    return {
        k: ann for k, ann in type(node).__annotations__.items()
        if is_loc_field(ann)
    }

def access_sites(node: Node, model: Model):
    sites = {
        k: getattr(node, k).site().access(model)
        for k, _ in loc_fields(node).items()
        if getattr(node, k) is not None
    }
    return sites

def access_locs(node: Node, model: Model):
    return {
        k: getattr(node, k).access(model)
        for k, _ in loc_fields(node).items()
        if getattr(node, k) is not None
    }

# ---------------------------------------------------------------

def check_node(node: Node, model: Model) -> bool:
    assert hasattr(node, "apply"), node
    assert hasattr(node, "init"), node
    return True

def add_node(model: Model, node: Node, **kws) -> Model_w_Location:
    assert check_node(node, model)
    i = len(model.sites)
    loc = Location.SITE(i)
    return Model_w_Location(
        model._replace(
            sites=model.sites.append(Site(
                node,
                loc=loc,
                **kws
            ))
        ), 
        loc,
        #
    )

# ---------------------------------------------------------------

def sweep_nodes(stages, ready, remaining, f_ready):
    ready_ = remaining.filterstar(
        lambda i, site: f_ready(site, ready)
    ).zip().map(xt.iTuple)
    if not len(ready_):
        return stages, ready, remaining, 0
    ready_i, ready_sites = ready_
    remaining = remaining.filterstar(
        lambda i, site: i not in ready_i
    )
    ready = ready.extend(ready_sites.map(lambda s: s.loc))
    stages = stages.append(ready_i)
    return stages, ready, remaining, ready_i.len()

def children_ready(children, site, ready):
    return children[site.loc].all(lambda loc: loc in ready)

def inputs_ready(children):
    def f_ready(site, ready):
        return (
            children_ready(children, site, ready)
            and site.input
            #
        )
    return f_ready

def static_ready(children):
    def f_ready(site, ready):
        return (
            children_ready(children, site, ready)
            and site.static
            #
        )
    return f_ready

def body_ready(children):
    def f_ready(site, ready):
        return (
            children_ready(children, site, ready)
            and not site.constraint
            #
        )
    return f_ready

def constraint_ready(children):
    def f_ready(site, ready):
        return (
            children_ready(children, site, ready)
            and site.constraint
            #
        )
    return f_ready

def order_nodes(model: Model) -> Model:
    # order: position in .results, sorted by position in .sites
    # groups: ordered groups of position in .sites

    children = {
        site.loc.site(): xt.iTuple.from_values(
            access_sites(site.node, model)
        ).map(lambda s: s.loc.site()).sort(
            lambda s: s.i
        ) for site in model.sites
    }

    remaining = model.sites.enumerate()

    stages = xt.iTuple()
    ready = xt.iTuple()

    stages, ready, remaining, _ = sweep_nodes(
        stages, ready, remaining, inputs_ready(children)
    )
    
    remaining_inputs = remaining.filter(
        lambda i_site: i_site[1].input
    )
    assert not remaining_inputs.len(), remaining_inputs

    for f in [
        static_ready(children),
        body_ready(children),
        constraint_ready(children),
    ]:
        n_change = None
        while n_change != 0:
            stages, ready, remaining, n_change = sweep_nodes(
                stages, ready, remaining, f
            )

    assert remaining.len() == 0, remaining

    stages_flat = stages.flatten()
    order = model.sites.len_range().map(
        lambda i: stages_flat.index_of(i)
    )
    
    return model._replace(
        order=order,
        stages=stages,
    )

# ---------------------------------------------------------------

def i_replace(model, sites, indices):
    i_sites = {
        i: site for i, site 
        in indices.zip(sites)
        #
    }
    return model._replace(
        sites=model.sites.enumerate().mapstar(
            lambda i, s: s if i not in i_sites else i_sites[i]
        )
    )


def init_model(model: Model, data: tuple) -> Model:

    model = order_nodes(model)

    def f_acc(
        model: Model,
        i: int,
        stage: xt.iTuple,
        data: tuple,
        #
    ) -> Model:
        if i == 0:
            if any(stage.map(lambda i: model.sites[i].input)):
                sites, params = stage.enumerate().mapstar(
                    lambda _i, s: model.sites[s].init(
                        model, data=data[_i]
                    )
                ).zip().map(xt.iTuple)
                return i_replace(model._replace(
                    params = model.params.extend(params)
                ), sites, stage)

        call_init = operator.methodcaller("init", model)
        sites, params = stage.map(
            lambda i: call_init(model.sites[i])
        ).zip().map(xt.iTuple)

        return i_replace(model._replace(
            params = model.params.extend(params)
        ), sites, stage)

    # stage = iTuple of indices in original model.sites
    model = model.stages.enumerate().foldstar(
        lambda acc, i, stage: f_acc(
            acc, i, stage, data
        ),
        initial=model._replace(params = xt.iTuple())
    )
    return model
       
# ---------------------------------------------------------------
 
def apply_flags(model: Model, **flags):
    return model._replace(
        sites=model.sites.map(
            lambda site: site.apply_flags(flags)
        )
    )

def n_stages_where_all(model, stages, f, offset = 0):
    n = stages[offset:].len_range().first_where(
        lambda s: not stages[offset + s].all(
            lambda i: f(model.sites[i])
        ),
    )
    n = 0 if n is None else n
    return n

def rec_detach(v):
    if v is None:
        return v
    elif isinstance(v, jax.numpy.ndarray):
        return jax.lax.stop_gradient(v)
    elif isinstance(v, numpy.ndarray):
        return v
    elif isinstance(v, tuple):
        return xt.iTuple(v).map(rec_detach)
    elif isinstance(v, xt.iTuple):
        return v.map(rec_detach)
    elif isinstance(v, float):
        return v
    else:
        assert False, v

def init_objective(
    model: Model,
    data,
    init_rand_keys,
    jit = True,
    markov: typing.Optional[dict] = None,
):

    n_inputs = n_stages_where_all(
        model,
        model.stages,
        lambda site: site.input,
        offset=0
    )
    assert n_inputs <= 1

    n_static = n_stages_where_all(
        model,
        model.stages,
        lambda site: site.static,
        offset=n_inputs
    )
    n_constraints = n_stages_where_all(
        model,
        model.stages.reverse(),
        lambda site: site.constraint,
        offset=0
    )

    model = model._replace(random=init_rand_keys)

    if n_inputs == 1:
        i_inputs = model.stages[0]
        assert i_inputs.map(lambda i: model.sites[i]).all(
            lambda site: site.input
        )
        init_results = i_inputs.enumerate().mapstar(
            lambda i, i_site: model.sites[i_site].apply(
                model, data=data[i]
            )
        )
        model = model._replace(results=init_results)

    def f_stage(res, i_stage):
        call_apply = operator.methodcaller("apply", res)
        res = i_stage.map(lambda i: call_apply(res.sites[i]))
        return res

    model = model.stages[n_inputs: n_inputs + n_static].fold(
        lambda res, i_stage: res._replace(
            results=res.results.extend(f_stage(res, i_stage))
        ),
        initial=model,
    )
    
    markov_sites = model.sites.filter(lambda s: s.markov)

    def f_res(params, rand_keys, **flags):
        res = model.stages[n_inputs + n_static:].fold(
            lambda acc, i_stage: acc._replace(
                results=acc.results.extend(f_stage(acc, i_stage))
            ),
            initial=model._replace(
                params = params,
                random=rand_keys,
            ),
        )
        loss = jax.numpy.stack(model.stages[-n_constraints:].map(
            lambda stage: stage.map(
                lambda i: model.sites[i].loc.result().access(res)
            )
        ).flatten().pipe(list)).sum()
        return res.results.pipe(tuple), loss

    if len(markov_sites) and markov is not None:
        if jit:
            f_res = jax.jit(f_res)

        def f(params, rand_keys, **flags):

            mask = get_markov_mask(
                markov, markov_sites, params, model.order
            )
            params = apply_markov_mask(mask, params)

            res, loss = f_res(params, rand_keys)

            if markov is not None and not jax.numpy.isnan(loss):
                _m = model._replace(results=xt.iTuple(res))
                for site in markov_sites:
                    markov[site.loc] = rec_detach(
                        site.loc.result().access(_m)
                    )

            return loss
        return f
    else:
        def f(params, rand_keys, **flags):
            _, loss = f_res(params, rand_keys)
            return loss
        if jit:
            f = jax.jit(f)
        return f

def apply_model(
    model,
    data,
    rand_keys=None,
    params = None,
    **flags,
):
    if params is None:
        params = model.params

    if rand_keys is None:
        rand_keys, _ = gen_rand_keys(model)

    model = model.apply_flags(**flags, apply = True)
    model = model._replace(random = rand_keys)

    n_inputs = n_stages_where_all(
        model,
        model.stages,
        lambda site: site.input,
        offset=0
    )
    assert n_inputs <= 1

    if n_inputs == 1:
        i_inputs = model.stages[0]
        assert i_inputs.map(lambda i: model.sites[i]).all(
            lambda site: site.input
        )
        init_results = i_inputs.enumerate().mapstar(
            lambda i, i_site: model.sites[i_site].apply(
                model, data=data[i]
            )
        )
        model = model._replace(results=init_results)

    def f_stage(res, i_stage):
        call_apply = operator.methodcaller("apply", res)
        return i_stage.map(lambda i: call_apply(res.sites[i]))

    model = model.stages[n_inputs:].fold(
        lambda res, i_stage: res._replace(
            results=res.results.extend(f_stage(res, i_stage))
        ),
        initial=model,
    )

    # ... acc

    return model

# ---------------------------------------------------------------

def gen_rand_keys(model: Model):
    ks = model.stages.map(
        lambda stage: stage.map(
            lambda i: (
                utils.rand.next_key()
                if model.sites[i].random
                else None
            )
        )
    ).flatten()
    n_keys = ks.filter(lambda v: v is not None).len()
    return ks.pipe(to_tuple_rec), n_keys

# ---------------------------------------------------------------

def get_markov_site(s, order):
    m = s.markov
    if isinstance(m, Location):
        loc = m
        return xt.iTuple((
            order[loc.i]
        ))
    elif isinstance(m, (tuple, xt.iTuple)):
        if isinstance(m, tuple):
            m = xt.iTuple(m)
        return m.not_none().assert_all(
            lambda loc: isinstance(loc, Location),
            f_error=lambda it: it.filter(
                lambda loc: not isinstance(loc, Location)
            )
        ).map(lambda loc: order[loc.i])
    else:
        assert False, s.markov

def get_markov_sites(sites, order):
    return sites.map(
        lambda s: get_markov_site(s, order)
    ).flatten()

def unpack_markov_res(loc, res, order, params):
    assert isinstance(loc, Location), loc
    i_param = order[loc.i]
    if loc.path.len() == 0:
        return i_param, res

    # TODO: if loc.path, only update the particular tuple element
    # ie. call replace on the relevant index in the original param element

    # in theory that means then re-merging the updated elements
    # if same loc.i

    assert False, loc

def get_markov_res(s, markov, order, params):
    m = s.markov
    res = markov[s.loc]
    if isinstance(m, Location):
        return xt.iTuple((
            unpack_markov_res(m, res, order, params),
        ))
    elif isinstance(m, (tuple, xt.iTuple)):
        if isinstance(m, tuple):
            m = xt.iTuple(m)
        assert len(m) == len(res), dict(m=len(m), res=len(res))
        return m.zip(res).filterstar(
            lambda loc, _res: loc is not None
        ).mapstar(
            lambda loc, _res: unpack_markov_res(
                loc, _res, order, params
            )
        )
    else:
        assert False, s.markov

def get_markov_mask(markov, sites, params, order):

    if sites.len() == 0:
        return {}

    mask_site = functools.partial(
        get_markov_res,
        markov=markov,
        order=order,
        params=params,
    )
    
    return {
        i: p for i, p in sites.map(mask_site).flatten()
    }

def apply_markov_mask(mask, params):
    return xt.iTuple(params).enumerate().mapstar(
        lambda i, p: p if i not in mask else mask[i]
    ).pipe(tuple)

# ---------------------------------------------------------------

def to_tuple_rec(v):
    if isinstance(v, xt.iTuple):
        return v.map(to_tuple_rec).pipe(tuple)
    return v

def has_nan(v):
    if isinstance(v, tuple):
        return any([has_nan(vv) for vv in v])
    return numpy.isnan(v).any()

def init_optimisation(
    model,
    data,
    jit=True,
    rand_init=0,
    **flags,
):
    
    test_loss = None
    params = objective = None

    score_model = model.apply_flags(**flags, init=True)

    rand_keys, _ = gen_rand_keys(model)

    objective = init_objective(
        score_model,
        data,
        init_rand_keys=rand_keys,
        jit = jit,
    )
    f_grad = jax.value_and_grad(objective)

    print("Rand init")

    tries = 0
    for iter in range(rand_init + 1):
        
        _params = (
            model.params
            if params is None
            else model.init(data).params
        ).pipe(to_tuple_rec)

        # TODO: don't re init input / static

        _test_loss, test_grad = f_grad(
            _params, rand_keys,
        )

        try:
            assert not has_nan(test_grad), (test_loss, test_grad,)
            if test_loss is None or _test_loss < test_loss:
                test_loss = _test_loss
                params = _params

            tries += 1
        except:
            if iter == rand_init:
                assert tries > 0

    return params

def optimise_model(
    model, 
    data,
    iters=1000,
    verbose=True,
    jit=True,
    rand_init=0,
    max_error_unchanged=None,
    lr = 0.01,
    opt=None,
    nan_termination=True,
    **flags,
):

    if max_error_unchanged is not None and max_error_unchanged < 1:
        max_error_unchanged *= iters

    if not model.sites.filter(lambda s: s.constraint).len():
        return model
    
    params = init_optimisation(
        model,
        data,
        rand_init=rand_init,
        jit=jit,
        **flags,
    )
    model = model._replace(params=params)

    train_model = model.apply_flags(**flags, train=True)
    
    rand_keys, _ = gen_rand_keys(model)

    markov_sites = model.sites.filter(lambda s: s.markov)
    markov_i = get_markov_sites(markov_sites, model.order)

    # init with the starting param values (of the masked site)
    if markov_sites.len():
        markov = {
            site.loc: rec_detach(
                model.sites[i].loc.param().access(model)
            )
            for i, site in markov_i.zip(markov_sites)
        }
    else:
        markov = None

    objective = init_objective(
        train_model,
        data,
        init_rand_keys=rand_keys,
        jit = jit,
        markov=markov,
    )

    if opt is None:
        opt = optax.adam(lr)
    
    # NOTE: use optax.sgd(1) for em algos

    solver = jaxopt.OptaxSolver(
        opt=opt, fun=objective, maxiter=iters, jit=jit
    )

    params = xt.iTuple(params).enumerate().mapstar(
        lambda i, p: p if i not in markov_i else ()
    ).pipe(tuple)
    state = solver.init_state(params) 

    error = None
    params_opt = None
    params_prev = None

    i_min = None
    error_min = None
    since_min = 0

    rand_keys, n_random = gen_rand_keys(model)

    for i in range(iters):
        params, state = solver.update(
            params,
            state,
            rand_keys,
        )
        error = state.error

        # TODO: markov needs to be kept at optimal as well?

        if numpy.isnan(error):
            if verbose: print("Hit NA, early termination")
            if not nan_termination:
                # we do not allow nan as a valid termination condition
                # so return prev params so we can debug what went wrong
                params = params_prev
            break

        params_prev = params

        if i % int(iters / 10) == 0:
            if verbose: print(i, error)
        
        if error_min is None or error < error_min:
            error_min = error
            since_min = 0
            params_opt = params
            i_min = i
        else:
            since_min += 1

        if (
            max_error_unchanged is not None 
            and since_min >= max_error_unchanged
        ):
            params = params_opt
            break

        if n_random > 0:
            rand_keys, _ = gen_rand_keys(model)

    if verbose:
        print(i, error)
        print("Min", i_min, error_min)

    if nan_termination:
        params = params_opt

    mask = get_markov_mask(
        markov, markov_sites, params, model.order
    )
    params = apply_markov_mask(mask, params)

    model = model._replace(params=params)
    return model

# ---------------------------------------------------------------
