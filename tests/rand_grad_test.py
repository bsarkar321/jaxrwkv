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
    model_choice: Literal[tuple(models.keys())] =  "4w0.1B"
    # n_layer: int = 3
    # n_embd: int = 256
    vocab_size: int = 10

    batch_size: int = 1
    sequence_length: int = 32
    new_start_prob: float = 0.1
    dtype: Optional[str] = None
    rwkv_type: str = "ScanRWKV"
    validation_rwkv_type: str = "ScanRWKV"


def construct_update_fn(forward_fn, include_state):
    def get_loss(params, tokens, full_state, lengths, new_starts):
        outs, state = forward_fn(params, tokens, full_state, lengths, new_starts)

        loss_out = jnp.mean(outs ** 2)
        loss_state = jnp.mean(jnp.clip(state ** 2, max=1.0))
        # logits = outs[:, :-1]
        # ans = tokens[:, 1:]
        # loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, ans))
        # return loss_out + loss_state
        # return loss_state * 1000
        return loss_out

    fast_loss = jax.jit(jax.value_and_grad(get_loss, argnums=(0, 2) if include_state else 0))
    # fast_loss = jax.jit(get_loss)

    return fast_loss


def print_tree(params, n=0):
    if not isinstance(params, dict):
        print(f":{params.shape} {params.dtype} {params.device}")
        return
    
    print("{")
    for k in params:
        print("\t" * (n+1) + k, end="")
        print_tree(params[k], n+1)
    print("\t" * n + "}")


def print_error_tree(params, o_params, n=0):
    if not isinstance(params, dict):
        if params > 3e-4:# and o_params > 1e-2:
            print('\033[93m', params, '\033[0m')
        else:
            print('\033[96m', params, '\033[0m')
        return
    
    print("{")
    for k in params:
        print("\t" * (n+1) + k, end="")
        print_error_tree(params[k], o_params[k], n+1)
    print("\t" * n + "}")
        
    
if __name__ == '__main__':
    args = tyro.cli(Args)
    rwkv_file = versions[args.version]
    VALID_RWKV = getattr(rwkv_file, args.validation_rwkv_type)
    RWKV, params, config, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)

    # RWKV, params, config = get_rand_model(args.seed, args.version, args.n_layer, args.n_embd, args.vocab_size, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)

    # print_tree(params)

    params = jax.device_put(params, jax.local_devices()[0])
    true_init_state = RWKV.default_state(params, config)

    forward = partial(RWKV.forward, config=config)


    _, init_state = jax.jit(forward)(params, jnp.array([0, 1, 2]), true_init_state)
    # init_state = true_init_state

    tok_key, len_key, ns_key = jax.random.split(jax.random.key(args.seed), 3)
    tokens = jax.random.randint(tok_key, (args.batch_size, args.sequence_length), 0, args.vocab_size)
    lengths = jax.random.randint(len_key, args.batch_size, 0, args.sequence_length) + 1
    new_starts = jax.random.uniform(ns_key, (args.batch_size, args.sequence_length)) < args.new_start_prob
    print("Number of new starts=", jnp.sum(new_starts))
    print("lengths", lengths)
    
    v_forward = jax.vmap(forward, in_axes=(None, 0, None, 0, 0))
    print("compiling")
    start_time = time.time()
    fast_loss_and_grad = construct_update_fn(v_forward, True).lower(params, tokens, init_state, lengths, new_starts).compile()
    print("compile time=", time.time() - start_time)

    loss, (grad_params, grad_state) = fast_loss_and_grad(params, tokens, init_state, lengths, new_starts)
    print(loss)
    print(jax.tree.map(lambda x: jnp.mean(jnp.abs(x)), grad_params))
    print([jnp.mean(jnp.abs(grad_state[:, x])) for x in range(grad_state.shape[1])])
    # import operator
    # print("grad signature", jax.tree.reduce(operator.add, jax.tree.map(lambda x: jnp.mean(jnp.abs(x)), grad)))



    val_forward = partial(VALID_RWKV.forward, config=config)
    val_v_forward = jax.vmap(val_forward, in_axes=(None, 0, None, 0, 0))
    print("compiling")
    start_time = time.time()
    val_fast_loss_and_grad = construct_update_fn(val_v_forward, True).lower(params, tokens, init_state, lengths, new_starts).compile()
    print("compile time=", time.time() - start_time)

    val_loss, (val_grad_params, val_grad_state) = val_fast_loss_and_grad(params, tokens, init_state, lengths, new_starts)
    print(val_loss)
    print(jax.tree.map(lambda x: jnp.mean(jnp.abs(x)), val_grad_params))
    print([jnp.mean(jnp.abs(val_grad_state[:, x])) for x in range(val_grad_state.shape[1])])
    print("*"*100)
    print("errors")
    print("loss", jnp.mean((loss - val_loss) ** 2))
    grad_state_diff = (grad_state - val_grad_state) ** 2
    print("grad state", [jnp.mean(jnp.abs(grad_state_diff[:, x])) for x in range(grad_state_diff.shape[1])])
    params_error = jax.tree.map(lambda x, y: jnp.mean(jnp.abs(x - y)), grad_params, val_grad_params)
    o_params_error = jax.tree.map(lambda x, y: jnp.mean(jnp.abs(x - y) / jnp.maximum(jnp.abs(y), 1.0)), grad_params, val_grad_params)
    # print("params", )
    print_error_tree(params_error, o_params_error)
