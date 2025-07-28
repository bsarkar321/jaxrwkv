import jax
import jax.numpy as jnp

import numpy as np

from dataclasses import dataclass, field
from functools import partialmethod

from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


@jax.tree_util.register_dataclass
@dataclass
class Partitioned:
    v: jax.Array
    idx: int = field(metadata=dict(static=True))
    axis_name: str = field(metadata=dict(static=True))

    def __getattr__(self, name):
        return getattr(gather_params(self), name) # pass through all attributes

def inner_fn(self, attr, *args, **kwargs):
    return getattr(gather_params(self), attr)(*args, **kwargs)

for attr in dir(jax.Array):
    if (attr not in dir(Partitioned) or attr in ['__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__']) and callable(getattr(jax.Array, attr)):
        setattr(Partitioned, attr, partialmethod(inner_fn, attr))
    
def shard_param(value, mesh, axis_name: str, min_weight_size: int = 2**18):
    if value.size < min_weight_size:
        return Partitioned(jax.device_put(value, NamedSharding(mesh, P())), 0, axis_name)
    axis_size = mesh.shape[axis_name]
    shape = value.shape
    idx = np.argsort(shape)[::-1]  # Shard along largest possible axis.
    for i in idx:
        if shape[i] % axis_size == 0:
            return Partitioned(jax.device_put(value, NamedSharding(mesh, P(*([None] * i + [axis_name])))), i-len(shape), axis_name)
    jax.debug.print("Skipping param due to incorrect size")
    return Partitioned(jax.device_put(value, NamedSharding(mesh, P())), 0, axis_name)

def gather_array_with_mean_grads(x: jax.Array, axis: int, axis_name: str):
    axis_size = jax.lax.psum(1, axis_name)
    @jax.custom_gradient
    def f(x):
        def grad_fn(g):
            return (jax.lax.psum_scatter(g, axis_name, scatter_dimension=axis, tiled=True) / axis_size)
        return jax.lax.all_gather(x, axis_name, axis=axis, tiled=True), grad_fn
    return f(x)

def gather_param(value):
    if value.idx == 0:
        return value.v
    else:
        return gather_array_with_mean_grads(value.v, axis=len(value.v.shape)+value.idx, axis_name=value.axis_name)

def gather_params(params):
    return jax.tree.map(lambda x: gather_param(x), params, is_leaf=lambda x: isinstance(x, Partitioned))

def sync_grads(value):
    if value.idx == 0:
        return Partitioned(jax.lax.pmean(value.v, value.axis_name), value.idx, value.axis_name)
    else:
        return value

def sharded_to_cpu(params):
    return jax.tree.map(lambda x: jax.device_put(x.v, jax.devices("cpu")[0]), params, is_leaf=lambda x: isinstance(x, Partitioned))
