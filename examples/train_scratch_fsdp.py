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
from dataclasses import dataclass, field
from typing import Literal

from jaxrwkv.auto import versions, get_rand_model
from jaxrwkv.utils.fsdp import Partitioned, shard_param, sync_grads, gather_params

from tqdm import trange

import time

from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, SingleDeviceSharding
from jax.sharding import PartitionSpec as P
    

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

    dp_axis_size: int = 1
    fsdp_axis_size: int = -1
    
def get_update_function(args, forward_fn, state_fn, solver, param_sharding, opt_sharding):

    def loss(params, tokens):
        print(tokens.shape)
        resets = (tokens == 0)
        logits, _ = forward_fn(params, tokens, state_fn(params), new_starts=resets if args.process_resets else None)
        do_xent = 1.0 - resets[1:]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits[:-1], tokens[1:])
        return jnp.sum(loss * do_xent) / jnp.sum(do_xent)

    def batch_loss(params, tokens):
        local_loss = jnp.mean(jax.vmap(jax.vmap(loss, in_axes=(None, 0)), in_axes=(None, 0))(params, tokens))
        return jax.lax.pmean(jax.lax.pmean(local_loss, "fsdp"), "dp")

    fast_batch_grad = jax.value_and_grad(batch_loss)

    @partial(shard_map, mesh=mesh, in_specs=(param_sharding, opt_sharding, P("dp", "fsdp", None)), out_specs=(param_sharding, opt_sharding, P()), check_rep=True)
    def do_update(params, optimizer, tokens_batch):
        loss, grad = fast_batch_grad(params, tokens_batch)
        grad = jax.lax.pmean(jax.tree.map(sync_grads, grad, is_leaf=lambda x: isinstance(x, Partitioned)), "dp")
        updates, optimizer = solver.update(grad, optimizer, params)
        params = optax.apply_updates(params, updates)
        return params, optimizer, loss

    return do_update
    

def main():
    global device_array, mesh
    args = tyro.cli(Args)

    device_array = np.array(jax.devices()).reshape((args.dp_axis_size, args.fsdp_axis_size))
    mesh = Mesh(device_array, ("dp", "fsdp"))
    

    assert args.batch_size % device_array.size == 0,  "batch size needs to be multiple of number of devices"

    print("creating initial model")
    RWKV, params, config = get_rand_model(args.seed, args.version, args.n_layer, args.n_embd, args.vocab_size, dtype=args.dtype, rwkv_type=args.rwkv_type, verbose=True)
    print("original params shape", jax.tree.map(lambda x: x.shape, params))
    # params = jax.device_put(params, jax.local_devices()[0])
    params = jax.tree.map(lambda x: shard_param(x, mesh, "fsdp"), params)
    print("sharded params shape", jax.tree.map(lambda x: x.v.shape, params, is_leaf=lambda x: isinstance(x, Partitioned)))
    
    print("Number of parameters:", jax.tree.reduce(lambda *x: sum(x), jax.tree.map(jnp.size, params)) / 1000000, "million")

    solver = optax.adamw(1.0)
    optimizer = solver.init(jax.tree.map(lambda p: p.copy(), params))
    optimizer = jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, P())) if isinstance(x.sharding, SingleDeviceSharding) else x, optimizer)
    param_sharding = jax.tree.map(lambda x: x.sharding.spec, params)
    opt_sharding = jax.tree.map(lambda x: x.sharding.spec, optimizer)

    print("getting dataset")
    ds = np.load(args.train_dataset, mmap_mode='r')
    print("dataset size:", ds.shape)
    total_batch_size = args.context_length * args.batch_size
    num_iters = ds.shape[0] // total_batch_size
    print("number of iterations:", num_iters)

    forward_fn = partial(RWKV.forward, config=config)
    state_fn = partial(RWKV.default_state, config=config)

    print("Compiling update")
    fsdp_dim_size = args.batch_size // args.dp_axis_size if args.dp_axis_size != -1 else 1
    train_ex = jax.device_put(ds[:total_batch_size].reshape((args.dp_axis_size, fsdp_dim_size, args.context_length)), NamedSharding(mesh, P("dp", "fsdp", None)))
    
    start_time = time.time()
    update_fn = jax.jit(get_update_function(args, forward_fn, state_fn, solver, param_sharding, opt_sharding), donate_argnums=(0, 1)).lower(params, optimizer, train_ex).compile()
    print("compile time", time.time() - start_time)
    print(f"Memory analysis with {args.batch_size}x{args.context_length}")
    print(update_fn.memory_analysis())

    for t in trange(num_iters):
        tok_batch = ds[t*total_batch_size:(t+1)*total_batch_size]
        params, optimizer, loss = update_fn(params, optimizer, jnp.reshape(tok_batch, (args.dp_axis_size, fsdp_dim_size, args.context_length)))
        if t % 100 == 0:
            print(f"iter {t}: {loss}")

if __name__ == "__main__":
    main()
