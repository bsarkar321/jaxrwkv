import jax
import os
from huggingface_hub.constants import HF_HOME

os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'

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
    validation_rwkv_type: str = "ScanRWKV"
    
    context: str = "The Eiffel tower is in the city of"
    answer: str = " Paris"


if __name__ == '__main__':
    args = tyro.cli(Args)
    rwkv_file = models[args.model_choice][0]
    VALID_RWKV = getattr(rwkv_file, args.validation_rwkv_type)

    RWKV, params, config, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)
    params = jax.device_put(params, jax.local_devices()[0])

    encoded = tokenizer.encode(args.context)

    forward = partial(RWKV.forward, config=config)
    valid_forward = partial(VALID_RWKV.forward, config=config)    
    init_state = RWKV.default_state(params, config)

    print("Baseline Results")
    true_out, true_state = valid_forward(params, encoded, init_state)
    true_out = true_out[len(encoded) - 1]
    true_soft_out = jax.nn.softmax(true_out)
    true_values, true_indices = jax.lax.top_k(true_soft_out, 10)
    print("*"*100)
    print("Baseline")
    for i in range(10):
        print(f"{true_values[i].item() * 100}%: {tokenizer.decode([true_indices[i].item()])}")
    print("*"*100)


    print("Modified Results")
    encoded_answer = tokenizer.encode(args.answer)
    
    full_context = encoded + encoded_answer + encoded
    print(tokenizer.decode(full_context))
    start_segments = jnp.zeros(len(full_context), dtype=jnp.bool)
    start_segments = start_segments.at[len(encoded) + len(encoded_answer)].set(True)

    out, state = forward(params, full_context, init_state, length=len(encoded), new_starts=start_segments)

    mid_out = out[len(encoded) - 1]
    mid_soft_out = jax.nn.softmax(mid_out)
    mid_values, mid_indices = jax.lax.top_k(mid_soft_out, 10)
    print("*"*100)
    print("Middle (should be same compuatation)")
    for i in range(10):
        print(f"{mid_values[i].item() * 100}%: {tokenizer.decode([mid_indices[i].item()])}")
    print("*"*100)

    
    end_out = out[len(full_context) - 1]
    end_soft_out = jax.nn.softmax(end_out)
    end_values, end_indices = jax.lax.top_k(end_soft_out, 10)
    print("End (should be the same due to resets)")
    for i in range(10):
        print(f"{end_values[i].item() * 100}%: {tokenizer.decode([end_indices[i].item()])}")
    print("*"*100)

    print("truncated vs true TVD:", 0.5 * jnp.sum(jnp.abs(mid_soft_out - true_soft_out)))
    print("full vs true TVD:", 0.5 * jnp.sum(jnp.abs(end_soft_out - true_soft_out)))

    print("Checking State")
    _, alt_state = forward(params, full_context, init_state, length=len(full_context), new_starts=start_segments)
    print("truncated vs true", jnp.mean(jnp.abs(true_state - state)))
    print("full vs true", jnp.mean(jnp.abs(true_state - alt_state)))
    

    
    
