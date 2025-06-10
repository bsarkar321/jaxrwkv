import jax
import jax.numpy as jnp

def batched_foo(a, b):
    return a * (b + 1)

@jax.custom_batching.custom_vmap
def foo(a, b):
    return batched_foo(a, b)

@foo.def_vmap
def foo_vmap_rule(axis_size, in_batched, a, b):
    print(axis_size, in_batched, a.shape, b.shape)
    return batched_foo(a, b), True

v_foo = jax.vmap(foo, in_axes=(1, 1), out_axes=1)
x = jnp.ones((10, 5))
y = jnp.ones((10, 5))
print(v_foo(x, y))

print(jax.make_jaxpr(v_foo)(x, y))
