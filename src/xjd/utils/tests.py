
import numpy

tolerances=dict(
    rtol=1e-3,
    atol=1e-2,
)

def assert_is_close(
    v1, v2, b=True, results=None, n_max=0, dp=3, **tols
):
    if results is None:
        results = dict(v1=numpy.round(v1, dp), v2=numpy.round(v2, dp))

    tols = {**tolerances, **tols}
    diff = numpy.subtract(v1, v2)

    n_diff = ((
        numpy.isclose(v1, v2, **tols)
    ) != b).sum()

    assert n_diff <= n_max, dict(
        **results,
        diff=numpy.round(diff, 3),
        n_diff=n_diff,
        diff_max=numpy.max(numpy.abs(diff))
    )
