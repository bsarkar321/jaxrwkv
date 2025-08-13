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

@dataclass
class Args:
    seed: int = 0
    model_choice: Literal[tuple(models.keys())] =  "7g0.1B"

    rwkv_type: str = "CudaRWKV"
    dtype: Optional[str] = None

    parallel_generations: int = 128

    warmup_iters: int = 100
    timing_iters: int = 100

    evo_sigma: float = 1e-5
    lora_dim: int = 1

    use_antithetic: bool = True

args = tyro.cli(Args)

if args.use_antithetic:
    assert args.parallel_generations % 2 == 0, "With antithetic generations, there should be even number of parallel generations"

RWKV, params, config, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)
params = jax.device_put(params, jax.local_devices()[0]) # move it to gpu (or whatever the default device is)
init_state = RWKV.default_state(params, config)
init_states = jnp.repeat(init_state[None], args.parallel_generations, axis=0)

params_shape = jax.tree.map(lambda x:x.shape, params)
print(params_shape)
print("mean parameter sizes")
print(jax.tree.map(lambda x:jnp.mean(jnp.abs(x)), params))

key = jax.random.key(args.seed)

key, master_evo_key = jax.random.split(key)
if args.use_antithetic:
    core_evo_keys = jnp.repeat(jax.random.split(master_evo_key, args.parallel_generations // 2), 2, axis=0)
    evo_keys, gen_keys = jax.vmap(jax.random.split, out_axes=1)(core_evo_keys)
    main_evo_sigma = args.evo_sigma * jnp.ones(args.parallel_generations // 2, dtype=init_state.dtype)
    sigma_antithetic = jnp.stack((main_evo_sigma, -main_evo_sigma), axis=-1).ravel()
else:
    core_evo_keys = jax.random.split(master_evo_key, args.parallel_generations)
    evo_keys, gen_keys = jax.vmap(jax.random.split, out_axes=1)(core_evo_keys)
    sigma_antithetic = args.evo_sigma * jnp.ones(args.parallel_generations, dtype=init_state.dtype)

def generate_model_keys(params, evo_key, sigma_antithetic):
    vals, treedef = jax.tree.flatten(params)
    all_keys = jax.random.split(evo_key, len(vals))
    partial_key_tree = jax.tree.unflatten(treedef, all_keys)
    n_layer = params['blocks']['att']['r_k'].shape[0]
    partial_key_tree['blocks'] = jax.tree.map(lambda x: jax.random.split(x, n_layer), partial_key_tree['blocks'])
    return jax.tree.map(lambda x, y: (x, jnp.ones(x.shape, dtype=sigma_antithetic.dtype) * sigma_antithetic), partial_key_tree, params)

# generate_model_structure(params, _key)
fast_generate_model_keys = jax.jit(jax.vmap(generate_model_keys, in_axes=(None, 0, 0)))

# print(evo_keys, sigma_antithetic)
model_keys = fast_generate_model_keys(params, evo_keys, sigma_antithetic)

print(jax.tree.map(lambda x:x.shape, model_keys))

# print("warmup")

# for x in tqdm.trange(args.warmup_iters):
#     model_structures = jax.block_until_ready(fast_generate_model_structure(params, evo_keys))

# print("actual key gen")
# total_time = 0
# for x in tqdm.trange(args.timing_iters):
#     start_time = time.time()
#     model_structures = jax.block_until_ready(fast_generate_model_structure(params, evo_keys))
#     total_time += time.time() - start_time
# print("Generation time (total):", (total_time / (args.timing_iters)))
# print("Generation time (per thread):", (total_time / (args.timing_iters * args.parallel_generations)))

def evo_lora2(M, param, key):
    # Replacement for M @ param; param is axb, M is cxa; need M @ params.T + (M @ A.T) @ B.T, so B is bxl and A is lxa
    a, b = param.shape
    lora_params = jax.random.normal(key[0], (a+b, args.lora_dim), dtype=param.dtype) * key[1]
    B = lora_params[:b]
    A = lora_params[b:].T
    return M @ param + (M @ A.T) @ B.T
    

def evo_lora(M, param, key):
    # Replacement for M @ param.T; param is axb, M is cxb; need M @ params.T + (M @ B) @ A, so B is bxl and A is lxa
    a, b = param.shape
    lora_params = jax.random.normal(key[0], (a+b, args.lora_dim), dtype=param.dtype) * key[1]
    B = lora_params[:b]
    A = lora_params[b:].T
    
    return M @ param.T + (M @ B) @ A
    

def evo(param, key):
    return param + jax.random.normal(key[0], param.shape, dtype=param.dtype) * key[1]


class EvoRWKV(RWKV):

    @classmethod
    def evo_channel_mixing_seq(cls, x, state, ffn, key_ffn, length, new_starts):
        sx = jnp.concatenate([state, x[:-1, :]], dtype=x.dtype)
        sx = jnp.where(new_starts[:, None], jnp.zeros_like(sx), sx)
        sx = sx - x
        xk = x + sx * evo(ffn['x_k'], key_ffn['x_k'])
        k = jnp.square(jax.nn.relu(evo_lora(xk, ffn['key']['weight'], key_ffn['key']['weight']))) # LORA
        return evo_lora(k, ffn['value']['weight'], key_ffn['value']['weight']), x[length - 1] # LORA

    @classmethod
    def evo_time_mixing_seq(cls, x, state, v_first, att, key_att, length, new_starts, H, S, layer_id):
        T, C = x.shape

        sx = jnp.concatenate([state[:1], x[:-1, :]], dtype=x.dtype)
        sx = jnp.where(new_starts[:, None], jnp.zeros_like(sx), sx)
        sx = sx - x

        xr = x + sx * evo(att['x_r'], key_att['x_r'])
        xw = x + sx * evo(att['x_w'], key_att['x_w'])
        xk = x + sx * evo(att['x_k'], key_att['x_k'])
        xv = x + sx * evo(att['x_v'], key_att['x_v'])
        xa = x + sx * evo(att['x_a'], key_att['x_a'])
        xg = x + sx * evo(att['x_g'], key_att['x_g'])

        r = evo_lora(xr, att['receptance']['weight'], key_att['receptance']['weight']) # LORA
        w = -jax.nn.softplus(-(evo(att['w0'], key_att['w0']) + evo_lora2(jnp.tanh(evo_lora2(xw, att['w1'], key_att['w1'])), att['w2'], key_att['w2']))) - 0.5 # LORA2, LORA2
        k = evo_lora(xk, att['key']['weight'], key_att['key']['weight']) # LORA
        v = evo_lora(xv, att['value']['weight'], key_att['value']['weight']) # LORA

        v_first = jnp.where(layer_id == 0, v, v_first)
        v = jnp.where(layer_id == 0, v, v + (v_first - v) * jax.nn.sigmoid(
            evo(att['v0'], key_att['v0']) + evo_lora2((evo_lora2(xv, att['v1'], key_att['v1'])), att['v2'], key_att['v2'])
        ))

        a = jax.nn.sigmoid(evo(att['a0'], key_att['a0']) + evo_lora2((evo_lora2(xa, att['a1'], key_att['a1'])), att['a2'], key_att['a2']))
        g = evo_lora2(jax.nn.sigmoid(evo_lora2(xg, att['g1'], key_att['g1'])), att['g2'], key_att['g2'])

        kk = k * evo(att['k_k'], key_att['k_k'])
        kk = kk.reshape(T, H, -1)
        kk = kk / jnp.maximum(jnp.linalg.norm(kk, axis=-1, keepdims=True), 1e-12)
        kk = kk.reshape(T, C)
        k = k * (1 + (a-1) * evo(att['k_a'], key_att['k_a']))

        state = state.at[0].set(x[length-1])
        s = jnp.reshape(state[1:, :], (H, S, S))

        r, w, k, v, a_i, b_i = tuple([val.reshape(T, H, S) for val in (r, w, k, v, -kk, kk * a)])

        state_new, out = cls.inner_loop(r, w, k, v, a_i, b_i, s, length, new_starts)
        state = state.at[1:].set(state_new.reshape(S, -1))
        x = out.reshape(T, H*S)

        x = group_norm(x, num_groups=H, weight=evo(att['ln_x']['weight'], key_att['ln_x']['weight']), bias=evo(att['ln_x']['bias'], key_att['ln_x']['bias']), eps = 64e-5)
        x = x + (jnp.sum(r.reshape(1, T, H, -1) * k.reshape(1, T, H, -1) * evo(att['r_k'], key_att['r_k']), axis=-1, keepdims=True) * v.reshape(1, T, H, -1)).reshape(T, C)
        x = x * g
        return evo_lora(x, att['output']['weight'], key_att['output']['weight']), state, v_first # LORA

    @classmethod
    def evo_forward_seq(cls, params, key_params, config, x, state, length, new_starts):
        n_layer = params['blocks']['att']['r_k'].shape[0]
        n_head, head_size = params['blocks']['att']['r_k'][0].shape
        x = layer_norm(x, jax.tree.map(evo, params['ln0'], key_params['ln0']))

        v_first = x

        @partial(jax.checkpoint,
                 policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        def block_loop(y, inputs):
            x, v_first = y
            block, key_block, state, idx = inputs
            x_new, s, v_first = cls.evo_time_mixing_seq(layer_norm(x, jax.tree.map(evo, block['ln1'], key_block['ln1'])), state[1:], v_first, block['att'], key_block['att'], length, new_starts, n_head, head_size, idx)
            state = state.at[1:].set(s)
            x = x + x_new
            
            x_new, s = cls.evo_channel_mixing_seq(layer_norm(x, jax.tree.map(evo, block['ln2'], key_block['ln2'])), state[:1], block['ffn'], key_block['ffn'], length, new_starts)
            state = state.at[0].set(s)
            x = x + x_new
            return (x, v_first), state

        (x, _), state = jax.lax.scan(block_loop, (x, v_first), (params['blocks'], key_params['blocks'], state, jnp.arange(n_layer)))
        return x, state

    @classmethod
    def evo_forward(cls, params, key_params, tokens, state, length=None, new_starts=None, config=None):
        """
        Forward pass on a single stream of tokens
        """
        tokens = jnp.array(tokens)
        x = cls.embed(params, config, tokens) # doesn't include key_params
        T, D = x.shape
        if length is None:
            length = T
        if new_starts is None:
            new_starts = jnp.zeros((T,), dtype=jnp.bool)
        x, state = cls.evo_forward_seq(params, key_params, config, x, state, length, new_starts)
        x = cls.outhead(params, config, x) # doesn't include key_params
        return x, state


# params, key_params, tokens, state, length
batch_forward = jax.jit(jax.vmap(partial(EvoRWKV.evo_forward, config=config), in_axes=(None, 0, 0, 0)))

def _forward_and_sample(model, model_keys, input_tokens, input_states, generation_key):
    print("RECOMPILE")
    gen_keys, _gen_keys = jax.vmap(jax.random.split, out_axes=1)(generation_key)
    generated_outs, generated_states = jax.block_until_ready(batch_forward(model, model_keys, input_tokens, init_states))
    sampled_toks = jax.vmap(jax.random.categorical)(_gen_keys, generated_outs[:, -1:])
    return sampled_toks, generated_states, gen_keys

print("Compiling")
start_time = time.time()
forward_and_sample = jax.jit(_forward_and_sample).lower(
    params, model_keys, jax.random.choice(key, params['emb']['weight'].shape[0], shape=(args.parallel_generations, 1)), init_states, jax.random.split(key, args.parallel_generations)
).compile()
print("Compile time", time.time() - start_time)
print("memory info")
print(forward_and_sample.memory_analysis())
    
print("doing warmup")
for x in tqdm.trange(args.warmup_iters):
    key, _key = jax.random.split(key)
    prompt = jax.random.choice(_key, params['emb']['weight'].shape[0], shape=(args.parallel_generations, 1))
    input_toks, input_states, _ = jax.block_until_ready(forward_and_sample(params, model_keys, prompt, init_states, jax.random.split(_key, args.parallel_generations)))

print("start timing")
total_time = 0
input_toks = jax.block_until_ready(jnp.zeros((args.parallel_generations, 1), dtype=jnp.int32))
input_states = init_states
for x in tqdm.trange(args.timing_iters):
    start_time = time.time()
    input_toks, input_states, gen_keys = jax.block_until_ready(forward_and_sample(params, model_keys, input_toks, input_states, gen_keys))
    total_time += time.time() - start_time
ans = args.timing_iters * args.parallel_generations / total_time
print("tokens per second:", ans)

with open("evo_times.txt", "a") as myfile:
    myfile.write(f"{args.parallel_generations}\t{ans}\n")
