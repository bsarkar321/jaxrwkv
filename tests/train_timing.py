import jax
import os
from huggingface_hub.constants import HF_HOME

jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "jaxrwkvcomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import jax.numpy as jnp
import numpy as np

from jaxrwkv import get_model, models

from functools import partial

import tyro

import time

from dataclasses import dataclass, field
from typing import Optional, Literal

import optax

@dataclass
class Args:
    seed: int = 0
    model_choice: Literal[tuple(models.keys())] =  "4w0.1B"

    batch_size: int = 1
    lengths_per_trial: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    dtype: Optional[str] = None
    rwkv_type: str = "ScanRWKV"

    num_trials: int = 3

    
def construct_update_fn(forward_fn):
    def get_loss(params, tokens, full_state):
        outs, state = forward_fn(params, tokens, full_state)

        logits = outs[:, :-1]
        ans = tokens[:, 1:]
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, ans))
        return loss
    
    fast_loss = jax.jit(jax.value_and_grad(get_loss))

    def do_update(params, optimizer, tokens, state):
        loss, grad = fast_loss(params, tokens, state)
        updates, optimizer = solver.update(grad, optimizer, params)
        params = optax.apply_updates(params, updates)
        return params, loss, optimizer

    return jax.jit(do_update)#, donate_argnums=(0, 1))

if __name__ == "__main__":
    args = tyro.cli(Args)
    RWKV, params, config, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)

    params = jax.device_put(params, jax.local_devices()[0])
    true_init_state = RWKV.default_state(params, config)

    forward = partial(RWKV.forward, config=config)
    v_forward = jax.jit(jax.vmap(forward, in_axes=(None, 0, None)))
    
    solver = optax.sgd(1e-4)
    optimizer = solver.init(jax.tree.map(lambda p: p.copy(), params))
    update_fn = construct_update_fn(v_forward)

    max_tokens = max(args.lengths_per_trial)
    tokens = jax.random.randint(jax.random.key(args.seed), (args.batch_size, max_tokens), 0, 1024)

    print("Compiling")
    for t in args.lengths_per_trial:
        start_time = time.time()
        print(t)
        _, loss, _ = jax.block_until_ready(update_fn(params, optimizer, tokens[:, :t], true_init_state))
        end_time = time.time()
        print("Compile time:", end_time-start_time)

    
    print("*"*100)
    print(f"Running @bs={args.batch_size}, avg over {args.num_trials}")
    for t in args.lengths_per_trial:
        start_time = time.time()
        # print(t)
        for trial in range(args.num_trials):
            _, loss, _ = jax.block_until_ready(update_fn(params, optimizer, tokens[:, :t], true_init_state))
        end_time = time.time()
        print(f"{t: <5}: {(end_time - start_time) / args.num_trials:0.6f} sec; {t * args.batch_size * args.num_trials /(end_time - start_time):0.2f} tok/sec")


    
