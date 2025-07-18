import jax
import os
from huggingface_hub.constants import HF_HOME

jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "jaxrwkvcomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp
import optax
from functools import partial

import numpy as np

import tyro
from dataclasses import dataclass
from typing import Literal

from jaxrwkv.auto import versions, get_rand_model

from tqdm import trange

import time

@dataclass
class Args:
    train_dataset: str = "minipile_train.npy"

    seed: int = 0
    version: Literal[tuple(versions.keys())] = "6"
    n_layer: int = 6
    n_embd: int = 512
    vocab_size: int = 65530
    dtype: str = "bfloat16"
    rwkv_type: str = "AssociativeScanRWKV"

    context_length: int = 1024
    batch_size: int = 1

    process_resets: bool = True
    
def get_update_function(args, forward_fn, state_fn, solver):

    def loss(params, tokens):
        # logits, _ = forward_fn(params, tokens, state_fn(params))
        # return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits[:-1], tokens[1:]))
        resets = (tokens == 0)
        logits, _ = forward_fn(params, tokens, state_fn(params), new_starts=resets if args.process_resets else None)
        do_xent = 1.0 - resets[1:]
        # jax.debug.print("x={x}", x=jnp.sum(do_xent))
        loss = optax.softmax_cross_entropy_with_integer_labels(logits[:-1], tokens[1:])
        # jax.debug.print("unmasked loss={loss}; masked loss={loss2}", loss=jnp.mean(loss), loss2=(loss * do_xent) / jnp.sum(do_xent))
        return jnp.sum(loss * do_xent) / jnp.sum(do_xent)

    def batch_loss(params, tokens):
        return jnp.mean(jax.vmap(loss, in_axes=(None, 0))(params, tokens))

    fast_batch_grad = jax.value_and_grad(batch_loss)

    def do_update(params, optimizer, tokens_batch):
        loss, grad = fast_batch_grad(params, tokens_batch)
        updates, optimizer = solver.update(grad, optimizer, params)
        params = optax.apply_updates(params, updates)
        return params, optimizer, loss

    return do_update
    

def main():
    args = tyro.cli(Args)

    print("creating initial model")
    RWKV, params, config = get_rand_model(args.seed, args.version, args.n_layer, args.n_embd, args.vocab_size, dtype=args.dtype, rwkv_type=args.rwkv_type, verbose=True)
    params = jax.device_put(params, jax.local_devices()[0])
    print("Number of parameters:", jax.tree.reduce(lambda *x: sum(x), jax.tree.map(jnp.size, params)) / 1000000, "million")

    solver = optax.contrib.dadapt_adamw(1.0)
    optimizer = solver.init(jax.tree.map(lambda p: p.copy(), params))

    print("getting dataset")
    ds = np.load(args.train_dataset, mmap_mode='r')
    print("dataset size:", ds.shape)
    total_batch_size = args.context_length * args.batch_size
    num_iters = ds.shape[0] // total_batch_size
    print("number of iterations:", num_iters)

    forward_fn = partial(RWKV.forward, config=config)
    state_fn = partial(RWKV.default_state, config=config)

    print("Compiling update")
    start_time = time.time()
    update_fn = jax.jit(
        get_update_function(args, forward_fn, state_fn, solver),
        donate_argnums=(0, 1)).lower(params, optimizer, jax.ShapeDtypeStruct((args.batch_size, args.context_length), jnp.dtype('int32'))).compile()
    print("compile time", time.time() - start_time)
    print(f"Memory analysis with {args.batch_size}x{args.context_length}")
    print(update_fn.memory_analysis())

    for t in trange(num_iters):
        tok_batch = ds[t*total_batch_size:(t+1)*total_batch_size]
        params, optimizer, loss = update_fn(params, optimizer, jnp.reshape(tok_batch, (args.batch_size, args.context_length)))
        if t % 100 == 0:
            print(f"iter {t}: {loss}")

if __name__ == "__main__":
    main()
