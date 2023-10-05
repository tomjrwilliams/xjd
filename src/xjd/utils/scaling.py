
import typing

import jax
from . import shapes

class Unit_Sum(typing.NamedTuple):

    axis: int

    def __call__(self, v: jax.numpy.ndarray):
        axis = self.axis
        v_sum = shapes.expand_dims(
            v.sum(axis=axis), axis, v.shape[axis]
        )
        return jax.numpy.divide(v, v_sum)