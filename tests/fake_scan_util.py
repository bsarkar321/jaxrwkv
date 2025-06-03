
def fake_scan(f, init, xs):
    carry = init
    ys = []
    blocks, state = xs
    for t in range(state.shape[0]):
        carry, y = f(carry, jax.tree.map(lambda z: z[t], xs))
        ys.append(y)
    return carry, jnp.stack(ys)
