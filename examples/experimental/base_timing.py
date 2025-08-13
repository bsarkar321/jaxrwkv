import jax
import os
from huggingface_hub.constants import HF_HOME

jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "jaxrwkvcomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp

from jaxrwkv import get_model, models

from functools import partial

import tyro
from dataclasses import dataclass

import tqdm

from typing import Optional, Literal

import time

@dataclass
class Args:
    seed: int = 0
    model_choice: Literal[tuple(models.keys())] =  "7g0.1B"

    rwkv_type: str = "CudaRWKV"
    dtype: Optional[str] = None

    parallel_generations: int = 128

    warmup_iters: int = 100
    timing_iters: int = 100    

args = tyro.cli(Args)
RWKV, params, config, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)
params = jax.device_put(params, jax.local_devices()[0]) # move it to gpu (or whatever the default device is)

init_state = RWKV.default_state(params, config)
init_states = jnp.repeat(init_state[None], args.parallel_generations, axis=0)

params_shape = jax.tree.map(lambda x:x.shape, params)
print(params_shape)

key = jax.random.key(args.seed)

key, master_evo_key = jax.random.split(key)
gen_keys = jax.random.split(master_evo_key, args.parallel_generations)

batch_forward = jax.jit(jax.vmap(partial(RWKV.forward, config=config), in_axes=(None, 0, 0))) # params, tokens, state, length

def _forward_and_sample(model, input_tokens, input_states, generation_key):
    print("RECOMPILE")
    gen_keys, _gen_keys = jax.vmap(jax.random.split, out_axes=1)(generation_key)
    generated_outs, generated_states = jax.block_until_ready(batch_forward(model, input_tokens, init_states))
    sampled_toks = jax.vmap(jax.random.categorical)(_gen_keys, generated_outs[:, -1:])
    return sampled_toks, generated_states, gen_keys

print("Compiling")
start_time = time.time()
forward_and_sample = jax.jit(_forward_and_sample).lower(
    params, jax.random.choice(key, params['emb']['weight'].shape[0], shape=(args.parallel_generations, 1)), init_states, jax.random.split(key, args.parallel_generations)
).compile()
print("Compile time", time.time() - start_time)
print("memory info")
print(forward_and_sample.memory_analysis())

print("doing warmup")
for x in tqdm.trange(args.warmup_iters):
    key, _key = jax.random.split(key)
    prompt = jax.random.choice(_key, params['emb']['weight'].shape[0], shape=(args.parallel_generations, 1))
    input_toks, input_states, _ = jax.block_until_ready(forward_and_sample(params, prompt, init_states, jax.random.split(_key, args.parallel_generations)))

print("start timing")
total_time = 0
input_toks = jnp.zeros((args.parallel_generations, 1), dtype=jnp.int32)
input_states = init_states
for x in tqdm.trange(args.timing_iters):
    start_time = time.time()
    input_toks, input_states, gen_keys = jax.block_until_ready(forward_and_sample(params, input_toks, input_states, gen_keys))
    total_time += time.time() - start_time
ans = args.timing_iters * args.parallel_generations / total_time
print("tokens per second:", ans)

with open("base_times.txt", "a") as myfile:
    myfile.write(f"{args.parallel_generations}\t{ans}\n")
