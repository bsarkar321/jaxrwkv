import jax
import os
from huggingface_hub.constants import HF_HOME

jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "jaxrwkvcomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp

from jaxrwkv import get_model, models
from jaxrwkv.rwkv7 import layer_norm, group_norm

from functools import partial

import tyro
from dataclasses import dataclass

import tqdm

from typing import Optional, Literal

import time
import wandb

import numpy as np

from simple_reward_functions import reward_functions # local import

import operator


@dataclass
class Args:
    seed: int = 0
    model_choice: Literal[tuple(models.keys())] =  "7g0.1B"

    rwkv_type: str = "CudaRWKV"
    dtype: Optional[str] = None

    parallel_generations: int = 1024
    generation_length: int = 100

    num_epochs: int = 100

    lr: float = 1e-4
    num_minibatches: int = 16
    clip_eps: float = 0.2
    # evo_sigma: float = 1e-3
    # lora_dim: int = 1

    # use_antithetic: bool = True

    reward_fn: Literal[tuple(reward_functions.keys())] = "fastzero"

    wandb_project: str = "evorwkv"
    wandb_name: str = "grpo"
    track: bool = False

    freeze_lora: bool = False
    freeze_nonlora: bool = False

args = tyro.cli(Args)

RWKV, params, config, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)

batch_forward = jax.jit(jax.vmap(partial(RWKV.forward, config=config), in_axes=(None, 0, 0)))

UNCHANGED = 0
FULL = 1
LORA = 2

lora_map = {'blocks': {
    'att': {'a0': FULL, 'a1': LORA, 'a2': LORA, 'g1': LORA, 'g2': LORA, 'k_a': FULL, 'k_k': FULL, 'key': {'weight': LORA},
            'ln_x': {'bias': FULL, 'weight': FULL}, 'output': {'weight': LORA},
            'r_k': FULL, # LORA EXCEPTION
            'receptance': {'weight': LORA},
            'v0': FULL, 'v1': LORA, 'v2': LORA,
            'value': {'weight': LORA},
            'w0': FULL, 'w1': LORA, 'w2': LORA, 'x_a': FULL, 'x_g': FULL, 'x_k': FULL, 'x_r': FULL, 'x_v': FULL, 'x_w': FULL},
    'ffn': {'key': {'weight': LORA}, 'value': {'weight': LORA}, 'x_k': FULL},
    'ln1': {'bias': FULL, 'weight': FULL}, 'ln2': {'bias': FULL, 'weight': FULL}},
    'emb': {'weight': UNCHANGED},
    'head': {'weight': UNCHANGED},
    'ln0': {'bias': FULL, 'weight': FULL},
    'ln_out': {'bias': FULL, 'weight': FULL}
}

def _forward_and_sample(model, input_tokens, input_states, generation_key):
    print("RECOMPILE")
    gen_keys, _gen_keys = jax.vmap(jax.random.split, out_axes=1)(generation_key)
    generated_outs, generated_states = jax.block_until_ready(batch_forward(model, input_tokens, input_states))
    sampled_toks = jax.vmap(jax.random.categorical)(_gen_keys, generated_outs[:, -1:])
    return sampled_toks, generated_states, gen_keys

def _generate_batch(params, gen_keys):
    input_toks = jax.block_until_ready(jnp.zeros((gen_keys.shape[0], 1), dtype=jnp.int32))
    def inner_scan(carry, inputs):
        toks, states, gen_keys = carry
        toks, states, gen_keys = _forward_and_sample(params, toks, states, gen_keys)
        return (toks, states, gen_keys), toks[:, 0]
    _, out_tokens = jax.lax.scan(inner_scan, (input_toks, init_states, gen_keys), length=args.generation_length)
    return out_tokens.T

batch_fitness = reward_functions[args.reward_fn]

def single_example_loss(params, old_params, tokens, advantage):
    T = tokens.shape[0]
    token_padding = (16 - T % 16) % 16
    input_tokens = jnp.concatenate((jnp.zeros_like(tokens[:1]), tokens[:-1], jnp.zeros_like(tokens[:token_padding])))
    pi, _ = RWKV.forward(params, input_tokens, RWKV.default_state(params, config), config=config)
    old_pi, _ = RWKV.forward(old_params, input_tokens, RWKV.default_state(params, config), config=config)

    pi_logprob = jax.nn.log_softmax(pi[:T])[jnp.arange(T), tokens]
    old_pi_logprob = jax.nn.log_softmax(old_pi[:T])[jnp.arange(T), tokens]
    ratio = jnp.exp(pi_logprob - old_pi_logprob)

    token_loss = -jnp.minimum(
        ratio * advantage,
        jnp.clip(ratio, 1-args.clip_eps, 1+args.clip_eps) * advantage
    )

    # pg_loss1 = -advantage * ratio
    # pg_loss2 = -advantage * jnp.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    # pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    return jnp.mean(token_loss)

def batch_loss(params, old_params, batch_tokens, advantages):
    return jnp.mean(jax.vmap(single_example_loss, in_axes=(None, None, 0, 0))(params, old_params, batch_tokens, advantages))

fast_batch_grad = jax.value_and_grad(batch_loss)

import optax

# def _do_update(params, model_keys, raw_scores, lr):
def _do_update(params, generations, raw_scores, lr):
    true_scores = (raw_scores - jnp.mean(raw_scores, keepdims=True)) / jnp.sqrt(jnp.var(raw_scores, keepdims=True) + 1e-5)

    solver = optax.adam(lr)
    optimizer = solver.init(params)
    
    def update_loop(state, x):
        new_params, optimizer = state
        gen_batch, score_batch = x
        loss, grad = fast_batch_grad(new_params, params, gen_batch, score_batch)
        updates, optimizer = solver.update(grad, optimizer, new_params)
        new_params = optax.apply_updates(new_params, updates)
        return (new_params, optimizer), loss
        # return jax.tree.map(lambda x, y, do_change: (x + lr * y * (do_change != UNCHANGED)).astype(x.dtype), new_params, grad, lora_map), loss
    (params, optimizer), losses = jax.lax.scan(update_loop, (params, optimizer), (generations.reshape(args.num_minibatches, args.parallel_generations // args.num_minibatches, -1), true_scores.reshape(args.num_minibatches, args.parallel_generations // args.num_minibatches)))
    # jax.debug.print("losses: {x}", x=losses)
    return params

params = jax.device_put(params, jax.local_devices()[0]) # move it to gpu (or whatever the default device is)
init_state = RWKV.default_state(params, config)
init_states = jnp.repeat(init_state[None], args.parallel_generations, axis=0)

key = jax.random.key(args.seed)
key, master_evo_key = jax.random.split(key)
gen_keys = jax.random.split(master_evo_key, args.parallel_generations)
# evo_keys, gen_keys, sigma_antithetic = generate_keys_from_master(master_evo_key)
# gen_keys = master_evo_key
# model_keys = fast_generate_model_keys(params, evo_keys, sigma_antithetic)

print("Compiling generate batch")
start_time = time.time()
generate_batch = jax.jit(_generate_batch).lower(params, gen_keys).compile()
print("Compile time", time.time() - start_time)
print("memory info")
print(generate_batch.memory_analysis())
print()
print("Compiling do update")
start_time = time.time()
do_update = jax.jit(_do_update).lower(params, jnp.zeros((args.parallel_generations, args.generation_length), dtype=jnp.int32), jnp.zeros(args.parallel_generations), args.lr).compile()
print("Compile time", time.time() - start_time)
print("memory info")
print(do_update.memory_analysis())

if args.track:
    run = wandb.init(
        project=args.wandb_project,
        config=args,
        name=args.reward_fn+args.wandb_name+f"_lr={args.lr}_bs={args.parallel_generations}"
    )
else:
    print("Run name:", args.reward_fn+args.wandb_name+f"_lr={args.lr}_bs={args.parallel_generations}")

for epoch in tqdm.trange(args.num_epochs):
    start_time = time.time()
    key, master_evo_key = jax.random.split(key)
    gen_keys = jax.random.split(master_evo_key, args.parallel_generations)
    # evo_keys, gen_keys, sigma_antithetic = generate_keys_from_master(master_evo_key)
    
    # # print(evo_keys, sigma_antithetic)
    # if epoch == 0:
    #     print("generating model keys")
    # model_keys = fast_generate_model_keys(params, evo_keys, sigma_antithetic)
    key_generation_time = time.time() - start_time

    start_time = time.time()
    if epoch == 0:
        print("generating batch")
    output_batch = jax.block_until_ready(generate_batch(params, gen_keys))
    token_generation_time = time.time() - start_time

    start_time = time.time()
    if epoch == 0:
        print("calculating fitness")
    output_scores = jax.block_until_ready(batch_fitness(output_batch, tokenizer))
    # print(np.array(output_scores).tolist())
    if epoch == 0:
        print("updating params")
    updated_params = jax.block_until_ready(do_update(params, output_batch, output_scores, args.lr))
    parameter_update_time = time.time() - start_time

    parameter_differences = jax.tree.map(lambda x, y:jnp.mean(jnp.abs(x-y)), params, updated_params)
    lora_updates = jax.tree.reduce(operator.add, jax.tree.map(lambda x, y: x if y == LORA else 0.0, parameter_differences, lora_map)) / jax.tree.reduce(operator.add, jax.tree.map(lambda y: 1.0 if y == LORA else 0.0, lora_map))
    nonlora_updates = jax.tree.reduce(operator.add, jax.tree.map(lambda x, y: x if y == FULL else 0.0, parameter_differences, lora_map)) / jax.tree.reduce(operator.add, jax.tree.map(lambda y: 1.0 if y == FULL else 0.0, lora_map))
    # print(jax.tree.map(lambda x, y:jnp.mean(jnp.abs(x-y)), params, updated_params))
    
    params = updated_params
    if args.track:
        run.log({
            "avg_fitness": jnp.mean(output_scores),
            "std_fitness": jnp.std(output_scores),
            "max_fitness": jnp.max(output_scores),
            "min_fitness": jnp.min(output_scores),
            "median_fitness": jnp.median(output_scores),
            "lora_updates": lora_updates,
            "nonlora_updates": nonlora_updates,
            "keygen_time": key_generation_time,
            "token_gen_time": token_generation_time,
            "update_time": parameter_update_time
        })
    else:
        print(f"Mean fitness: {jnp.mean(output_scores)}; std fitness: {jnp.std(output_scores)}; max fitness: {jnp.max(output_scores)}; min fitness: {jnp.min(output_scores)}; median fitness: {jnp.median(output_scores)}")
        print("mean parameter diffs")
        print("Lora modules:", lora_updates)
        print("Full modules:", nonlora_updates)

if args.track:
    run.finish()
