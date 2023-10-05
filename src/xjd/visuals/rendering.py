
import inspect

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

import IPython

import tempfile

import copy
import matplotlib
import matplotlib.cm

import functools

import numpy
import pandas

from .. import utils
from . import formatting

import xtuples as xt

# ---------------------------------------------------------------

# imports.import_from_file(
#     "xtuples", "c:/hc/xtuples/src/xtuples/__init__.py"
# )

# ---------------------------------------------------------------

def render_html(html):
    IPython.display.display(IPython.core.display.HTML(html))

# ---------------------------------------------------------------

def plotly_to_html(fig):
    with tempfile.NamedTemporaryFile(
        mode="w+"
        #
    ) as f_temp:
        fig.write_html(f_temp, include_plotlyjs="cdn")
        f_temp.seek(0)
        html = f_temp.read()
    return html

def render_as_html(fig):
    render_html(plotly_to_html(fig))

# ---------------------------------------------------------------

import re

def render_source(f, cls_method = False, until = None):
    code = inspect.getsource(f)
    if cls_method:
        name = f.__name__
        candidates = [loc.end() for loc in re.finditer(
            "@classmethod", code
        )]
        target = "def{}".format(name)
        for candidate in candidates:
            if target in (
                code[candidate:].replace(" ", "")
                .replace("\n", "")
                .replace("\t", "")
            ):
                code = "@classmethod {}".format(
                    formatting.unindent(code[candidate:])
                )
                break
    if until is not None:
        code = code.split(until)[0].strip()
    html = highlight(code, PythonLexer(), HtmlFormatter())
    formatter = HtmlFormatter()
    render_html(
        '<style type="text/css">{}</style>{}'.format(
            formatter.get_style_defs(".highlight"), html
        )
    )

# ---------------------------------------------------------------


def render_df_color_range(
    df,
    cmap="RdYlGn",
    dp=2,
    v_min=None,
    v_max=None,
    with_cols=None,
    excl_cols=None,
    symmetrical=None,
):

    cmap = copy.copy(matplotlib.cm.get_cmap(cmap))
    cmap.set_under("grey")
    
    if v_min is None or v_max is None:
        symmetrical = (
            symmetrical 
            if symmetrical is not None
            else (v_min is None and v_max is None)
            #
        )
        min_max_ks = dict(
            with_cols=with_cols,
            excl_cols=excl_cols,
            symmetrical=symmetrical,
        )
        if v_min is None and v_max is None:
            v_min, v_max = utils.dfs.df_min_max(df, **min_max_ks)
        elif v_min is None:
            v_min, _ = utils.dfs.df_min_max(df, **min_max_ks)
        elif v_max is None:
            _, v_max = utils.dfs.df_min_max(df, **min_max_ks)
        else:
            assert False, dict(v_min=v_min, v_max=v_max)

    border = (
        "border",
        "1px solid black",
    )

    def getlen(v):
        if dp == 0:
            return len(v.split(".")[0])
        return len(v)

    v_len = max(
        xt.iTuple(numpy.round(df.values, dp))
        .flatten()
        .map(str)
        .map(
            lambda v: (
                len(v) if dp != 0 else len(v.split(".")[0])
            )
        )
    )

    df = df.fillna(0)

    df = (
        df.style.background_gradient(
            axis=None,
            vmin=v_min,
            vmax=v_max,
            cmap=cmap,
        )
        .format(functools.partial(
            formatting.pad, v_len=v_len, dp=dp, 
        ))
        .set_properties(**{"font-size": "8pt"})
        .set_table_styles(
            [
                {"selector": "th", "props": [("font-size", "10pt")]},
                {"selector": "tbody td", "props": [border]},
                {"selector": "th", "props": [border]},
                dict(
                    selector=".row_heading",
                    props=[
                        (
                            "width",
                            "150px",
                        ),
                        (
                            "min-width",
                            "150px",
                        ),
                        (
                            "max-width",
                            "150px",
                        ),
                    ],
                ),
                dict(
                    selector=".col_heading",
                    props=[
                        (
                            "max-width",
                            "150px",
                        ),
                    ],
                ),
                #
            ]
        )
    )
    
    return df