
import functools
import datetime

import xtuples as xt

def y(y, m = 1):
    return datetime.date(y, m, 1)

def iter_dates(d, n = None, it = 1):
    if n is None:
        def f():
            step = datetime.timedelta(days=it)
            yield d
            while True:
                d += step
                yield d
    else:
        def f():
            return (
                xt.iTuple.range(n)
                .map(lambda i: d + datetime.timedelta(days=i * it))
            )
    return f()

starting = functools.partial(iter_dates, it = 1)
ending = functools.partial(iter_dates, it = -1)

def between(d1, d2):
    if d2 > d1:
        return starting(d1, (d2 - d1).days)
    return starting(d2, (d1 - d2).days).reverse()

import pandas

def date_index(d):
    if isinstance(d, dict):
        data=list(d.keys())
    elif isinstance(d, pandas.Series):
        data = list(d.index)
    else:
        data = list(d)
    return pandas.DatetimeIndex(data=data)

def dated_series(dd):
    return pandas.Series(
        data=list(dd.values()),
        index=date_index(dd.keys()),
    )