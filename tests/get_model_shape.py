import jax
import os
from huggingface_hub.constants import HF_HOME

jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "jaxrwkvcomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp
import numpy as np

from jaxrwkv import get_rand_model, get_model, models

from functools import partial

import tyro

import time

from dataclasses import dataclass
from typing import Optional, Literal

import json

import math

@dataclass
class Args:
    model_choice: Literal[tuple(models.keys())] =  "4w0.1B"

    dtype: Optional[str] = "float32"
    rwkv_type: str = "AssociativeScanRWKV"

    num_sequences: int = 1

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

    RWKV, params, config, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)
    params = jax.device_put(params, jax.local_devices()[0])

    print_tree(params)

    
