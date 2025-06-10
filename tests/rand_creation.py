import jax
import os
from huggingface_hub.constants import HF_HOME

jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "jaxrwkvcomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

jax.config.update('jax_default_matmul_precision', 'highest')

import jax.numpy as jnp
import numpy as np

from jaxrwkv import get_rand_model, get_model, models, versions

from functools import partial

import tyro

import time

from dataclasses import dataclass
from typing import Optional, Literal

import optax

@dataclass
class Args:
    seed: int = 0
    version: Literal[tuple(versions.keys())] = "4"
    n_layer: int = 3
    n_embd: int = 256
    vocab_size: int = 10

    dtype: Optional[str] = None
    rwkv_type: str = "ScanRWKV"

def print_tree(params, n=0):
    if not isinstance(params, dict):
        print(f":{params.shape} {params.dtype} {params.device}")
        return
    
    print("{")
    for k in params:
        print("\t" * (n+1) + k, end="")
        print_tree(params[k], n+1)
    print("\t" * n + "}")

if __name__ == '__main__':
    args = tyro.cli(Args)
    rwkv_file = versions[args.version]

    RWKV, params, config = get_rand_model(args.seed, args.version, args.n_layer, args.n_embd, args.vocab_size, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)

    params = jax.device_put(params, jax.local_devices()[0])

    print("parameter tree")
    print_tree(params)
    
    true_init_state = RWKV.default_state(params, config)

    forward = partial(RWKV.forward, config=config)

    print("Doing random forward pass")


    outs, state = forward(params, jnp.array([0, 1, 2]), true_init_state)
    print("All done")
