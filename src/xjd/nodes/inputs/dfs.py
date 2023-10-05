
from __future__ import annotations
import enum

import operator
import collections
# import collections.abc
import functools
import itertools

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

from ... import xfactors as xf
from ... import utils

# ---------------------------------------------------------------

# TODO: if eg. learning factor path over specific dates
# then here is where we encode that restriction
# specific stock universe, etc.


@xt.nTuple.decorate()
class DataFrame_Wide(typing.NamedTuple):

    allow_missing_columns: bool = True
    allow_missing_indices: bool = True

    allow_new_columns: bool = True
    allow_new_indices: bool = True

    na_threshold_columns: typing.Optional[float] = None
    na_threshold_indices: typing.Optional[float] = None

    given_columns: typing.Optional[xt.iTuple] = None
    given_index: typing.Optional[xt.iTuple] = None

    drop_na_columns: bool = True
    drop_na_rows: bool = True

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[DataFrame_Wide, tuple, xf.SiteValue]:
        assert site.loc is not None
        # path[0] = stage, so path[1] = index of data element

        df = data
        assert isinstance(df, pandas.DataFrame), df

        if self.drop_na_columns:
            df = df.dropna(axis=1, how = "all")
        if self.drop_na_rows:
            df = df.dropna(axis=0, how = "all")

        self = self._replace(given_columns = xt.iTuple(df.columns))
        self = self._replace(given_index = xt.iTuple(df.index))

        return self, df.values.shape, ()
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None

        df = data
        assert isinstance(df, pandas.DataFrame), df

        if self.drop_na_columns:
            df = df.dropna(axis=1, how = "all")
        if self.drop_na_rows:
            df = df.dropna(axis=0, how = "all")

        df = utils.dfs.apply_na_threshold(
            df,
            na_threshold=(
                self.na_threshold_columns,
                self.na_threshold_indices,
            ),
        )
        df = utils.dfs.apply_allow_missing(
            df,
            self.given_columns,
            self.given_index,
            self.allow_missing_columns,
            self.allow_missing_indices,
        )
        df = utils.dfs.apply_allow_new(
            df,
            self.given_columns,
            self.given_index,
            self.allow_new_columns,
            self.allow_new_indices,
        )

        return jax.numpy.array(df.values)

# ---------------------------------------------------------------

def rolling_dataframes(df, step, window, unit):
    
    assert window >= 1

    index_l = xt.iTuple(
        df.resample("{}{}".format(step, unit), label="left")
        .first()
        .index.values
    )
    index_r = xt.iTuple(
        df.resample("{}{}".format(step, unit), label="right")
        .last()
        .index.values
    )

    inds = index_l.zip(
        index_l.pretend(
            tuple([None for _ in range(window - 1)])
        ),
        index_r,
    )

    dfs = inds.mapstar(
        lambda l, start, r: df.loc[
            (df.index >= (
                l if start is None else start
            )) & (df.index <= r)
        ]
    )

    return dfs

@xt.nTuple.decorate()
class DataFrame_Wide_Rolling(typing.NamedTuple):

    step: int
    window: int
    unit: str

    allow_missing_columns: bool = True
    allow_missing_indices: bool = True

    allow_new_columns: bool = True
    allow_new_indices: bool = True

    na_threshold_columns: typing.Optional[float] = None
    na_threshold_indices: typing.Optional[float] = None

    given_columns: typing.Optional[xt.iTuple] = None
    given_index: typing.Optional[xt.iTuple] = None

    drop_na_columns: bool = True
    drop_na_rows: bool = True

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[DataFrame_Wide_Rolling, tuple, xf.SiteValue]:
        assert site.loc is not None
        # path[0] = stage, so path[1] = index of data element

        df = data
        assert isinstance(df, pandas.DataFrame), df

        if self.drop_na_columns:
            df = df.dropna(axis=1, how = "all")
        if self.drop_na_rows:
            df = df.dropna(axis=0, how = "all")

        dfs = (
            utils.dfs.rolling_windows(df, "{}{}".format(
                self.window, self.unit
            ), step = self.step)
            .map(
                functools.partial(
                    utils.dfs.apply_na_threshold,
                    na_threshold_columns=self.na_threshold_columns,
                    na_threshold_indices=self.na_threshold_indices,
                )
            )
        )

        self = self._replace(given_columns = dfs.map(
            lambda _df: xt.iTuple(_df.columns)
        ))
        self = self._replace(given_index = dfs.map(
            lambda _df: xt.iTuple(_df.index)
        ))

        return self, dfs.map(lambda _df: _df.values.shape), ()
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None

        df = data
        assert isinstance(df, pandas.DataFrame), df

        if self.drop_na_columns:
            df = df.dropna(axis=1, how = "all")
        if self.drop_na_rows:
            df = df.dropna(axis=0, how = "all")

        dfs = utils.dfs.rolling_windows(df, "{}{}".format(
            self.window, self.unit
        ), step = self.step)

        dfs = dfs.map(
            functools.partial(
                utils.dfs.apply_na_threshold,
                na_threshold_columns=self.na_threshold_columns,
                na_threshold_indices=self.na_threshold_indices,
            )
        ).map(
            functools.partial(
                utils.dfs.apply_allow_missing,
                allow_missing_columns=self.allow_missing_columns,
                allow_missing_indices=self.allow_missing_indices,
            ),
            self.given_columns,
            self.given_index,
        ).map(
            functools.partial(
                utils.dfs.apply_allow_new,
                allow_new_columns=self.allow_new_columns,
                allow_new_indices=self.allow_new_indices,
            ),
            self.given_columns,
            self.given_index,
        ).map(lambda _df: jax.numpy.array(_df.values))

        return dfs

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xf.init_null)
class Slice_DataFrame_Wide_Rolling_Columns(typing.NamedTuple):

    rolling: xf.Location
    slicing: xf.Location

    T: bool = False

    scale: typing.Optional[typing.Callable] = None

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[DataFrame_Tall, tuple, xf.SiteValue]: ... 
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
    
        slicing = self.slicing.access(state)

        rolling_columns = (
            self.rolling.site()
            .access(state)
            .node.given_columns
        )
        slicing_columns = (
            self.slicing.site()
            .access(state)
            .node.given_columns
        )

        vs = rolling_columns.map(
            lambda cs: slicing_columns.enumerate().filterstar(
                lambda i, c: c in cs
            ).mapstar(lambda i, c: i)
        ).map(
            lambda inds: slicing[..., inds.pipe(list)]
        )

        if self.scale is not None:
            vs = vs.map(self.scale)

        if self.T:
            return vs.map(lambda v: v.T)
        return vs

# ---------------------------------------------------------------

@xt.nTuple.decorate()
class DataFrame_Tall(typing.NamedTuple):

    # fields to specify if keep index and ticker map

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[DataFrame_Tall, tuple, xf.SiteValue]:
        assert site.loc is not None
        # path[0] = stage, so path[1] = index of data element
        return self, data[site.loc.path[1]].values.shape, ()
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        df = data
        assert isinstance(df, pandas.DataFrame), df
        return jax.numpy.array(df.values)

# ---------------------------------------------------------------
