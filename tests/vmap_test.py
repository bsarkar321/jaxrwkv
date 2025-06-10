import jax
import jax.numpy as jnp
import numpy as np

from jaxrwkv import get_model, models

from functools import partial

import tyro

import time

from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class Args:
    seed: int = 0
    model_choice: Literal[tuple(models.keys())] =  "4w0.1B"

    batch_size: int = 1
    dtype: Optional[str] = None
    rwkv_type: str = "ScanRWKV"
    
if __name__ == '__main__':
    args = tyro.cli(Args)
    RWKV, params, config, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)

    params = jax.device_put(params, jax.local_devices()[0])
    init_state = RWKV.default_state(params, config)

    context = "The Eiffel tower is in the city of"
    answer = "Paris"
    encoded = jnp.array(tokenizer.encode(context))

    forward = partial(RWKV.forward, config=config)

    ctxlen = jnp.size(encoded)

    v_forward = jax.jit(jax.vmap(forward, in_axes=(None, 0, 0, 0)))

    
    if isinstance(init_state, tuple):
        full_state = tuple([jnp.repeat(s[None], args.batch_size, axis=0) for s in init_state])
    else:
        full_state = jnp.repeat(init_state[None], args.batch_size, axis=0)

    full_tokens = jnp.repeat(encoded[None], args.batch_size, axis=0)
    full_length = jnp.ones(args.batch_size, dtype=full_tokens.dtype) * jnp.size(encoded)

    start_time = time.time()
    _, _ = jax.block_until_ready(v_forward(params, full_tokens, full_state, full_length))
    end_time = time.time()
    print(f"Compile time: {end_time - start_time} seconds")

    
    start_time = time.time()
    all_outs, all_states = jax.block_until_ready(v_forward(params, full_tokens, full_state, full_length))
    end_time = time.time()
    print(f"Run time: {end_time - start_time} seconds")

    outs = all_outs[:, jnp.size(encoded) - 1]
    soft_outs = jax.nn.softmax(outs)

    soft_out = soft_outs[0]
    values, indices = jax.lax.top_k(soft_out, 10)
    for i in range(10):
        print(f"{values[i].item() * 100}%: {tokenizer.decode([indices[i].item()])}")

    old_soft_out = soft_out

    print(f"State fingerprint: {jnp.mean(all_states[0])}")

    print("outs difference", jnp.mean(jnp.abs(all_outs - all_outs[:1])))
    print("states difference", jnp.mean(jnp.abs(all_states - all_states[:1])))


