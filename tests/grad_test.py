import jax
import jax.numpy as jnp
import numpy as np

from jaxrwkv import get_model, models

from functools import partial

import tyro

import time

from dataclasses import dataclass
from typing import Optional, Literal

import optax

@dataclass
class Args:
    seed: int = 0
    model_choice: Literal[tuple(models.keys())] =  "4w0.1B"

    batch_size: int = 1
    sequence_length: int = 32
    dtype: Optional[str] = None
    rwkv_type: str = "ScanRWKV"

    reset_point: int = 0


def construct_update_fn(forward_fn):
    def get_loss(params, tokens, full_state, lengths, new_starts):
        outs, state = forward_fn(params, tokens, full_state, lengths, new_starts)

        logits = outs[:, :-1]
        ans = tokens[:, 1:]
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, ans))
        return loss

    fast_loss = jax.jit(jax.value_and_grad(get_loss))
    # fast_loss = jax.jit(get_loss)

    return fast_loss


def print_tree(params, n=0):
    if not isinstance(params, dict):
        print(f":{params.shape}")
        return
    
    print("{")
    for k in params:
        print("\t" * (n+1) + k, end="")
        print_tree(params[k], n+1)
    print("\t" * n + "}")
        
    
if __name__ == '__main__':
    args = tyro.cli(Args)
    RWKV, params, config, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)

    print_tree(params)

    params = jax.device_put(params, jax.local_devices()[0])
    init_state = RWKV.default_state(params, config)
    
    if isinstance(init_state, tuple):
        full_state = tuple([jnp.repeat(s[None], args.batch_size, axis=0) for s in init_state])
    else:
        full_state = jnp.repeat(init_state[None], args.batch_size, axis=0)

    tokens = jnp.ones((args.batch_size, args.sequence_length), dtype=jnp.int32)
    lengths = jnp.ones((args.batch_size), dtype=jnp.int32) * args.sequence_length
    new_starts = jnp.zeros((args.batch_size, args.sequence_length), dtype=jnp.bool)
    new_starts = new_starts.at[:, args.reset_point].set(True)

    
    forward = partial(RWKV.forward, config=config)
    v_forward = jax.vmap(forward, in_axes=(None, 0, 0, 0, 0))
    # print(jax.make_jaxpr(construct_update_fn(v_forward)))
    print("compiling")
    start_time = time.time()
    fast_loss_and_grad = construct_update_fn(v_forward).lower(params, tokens, full_state, lengths, new_starts).compile()
    print("compile time=", time.time() - start_time)

    loss, grad = fast_loss_and_grad(params, tokens, full_state, lengths, new_starts)
    print(loss)
    import operator
    print("grad signature", jax.tree.reduce(operator.add, jax.tree.map(lambda x: jnp.mean(jnp.abs(x)), grad)))
