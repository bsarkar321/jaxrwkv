import jax
import os
from huggingface_hub.constants import HF_HOME

jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "jaxrwkvcomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp

from jaxrwkv import get_model, models
from jaxrwkv.rwkv7 import layer_norm, group_norm
from jaxrwkv.tokenizer import LegacyWorldTokenizer

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
    generation_length: int = 1000

    num_epochs: int = 100

    lr: float = 1e-4
    evo_sigma: float = 1e-3
    lora_dim: int = 1

    use_antithetic: bool = True

    wandb_project: str = "evorwkv"
    wandb_name: str = "full"
    track: bool = False

    freeze_lora: bool = False
    freeze_nonlora: bool = False

    # math specific params
    generations_per_prompt: int = 8

args = tyro.cli(Args)

assert args.parallel_generations % args.generations_per_prompt == 0, "The number of generations per prompt should evenly divide into parallel generations"

args.prompts_per_epoch = args.parallel_generations // args.generations_per_prompt


RWKV, params, config, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)
legacy_tokenizer = LegacyWorldTokenizer()

if args.use_antithetic:
    assert args.parallel_generations % 2 == 0, "With antithetic generations, there should be even number of parallel generations"


###### Evolution Model Implementation

def evo_lora2(M, param, key):
    if args.freeze_lora:
        return M@param
    # Replacement for M @ param; param is axb, M is cxa; need M @ params.T + (M @ A.T) @ B.T, so B is bxl and A is lxa
    a, b = param.shape
    lora_params = jax.random.normal(key[0], (a+b, args.lora_dim), dtype=param.dtype) * key[1]
    B = lora_params[:b]
    A = jnp.ones_like(lora_params[b:].T)
    return M @ param + (M @ A.T) @ B.T
    

def evo_lora(M, param, key):
    if args.freeze_lora:
        return M@param.T
    # Replacement for M @ param.T; param is axb, M is cxb; need M @ params.T + (M @ B) @ A, so B is bxl and A is lxa
    a, b = param.shape
    lora_params = jax.random.normal(key[0], (a+b, args.lora_dim), dtype=param.dtype) * key[1]
    B = lora_params[:b]
    A = jnp.ones_like(lora_params[b:].T)
    
    return M @ param.T + (M @ B) @ A
    

def evo(param, key):
    if args.freeze_nonlora:
        return param
    return param + jax.random.normal(key[0], param.shape, dtype=param.dtype) * key[1] # r_k is exception


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


def generate_model_keys(params, evo_key, sigma_antithetic):
    vals, treedef = jax.tree.flatten(params)
    all_keys = jax.random.split(evo_key, len(vals))
    partial_key_tree = jax.tree.unflatten(treedef, all_keys)
    n_layer = params['blocks']['att']['r_k'].shape[0]
    partial_key_tree['blocks'] = jax.tree.map(lambda x: jax.random.split(x, n_layer), partial_key_tree['blocks'])
    return jax.tree.map(lambda x, y: (x, jnp.ones(x.shape, dtype=sigma_antithetic.dtype) * sigma_antithetic), partial_key_tree, params)

fast_generate_model_keys = jax.jit(jax.vmap(generate_model_keys, in_axes=(None, 0, 0)))

batch_forward = jax.jit(jax.vmap(partial(EvoRWKV.evo_forward, config=config), in_axes=(None, 0, 0, 0)))

def _forward_and_sample(model, model_keys, input_tokens, input_states, generation_key):
    print("RECOMPILE")
    gen_keys, _gen_keys = jax.vmap(jax.random.split, out_axes=1)(generation_key)
    generated_outs, generated_states = jax.block_until_ready(batch_forward(model, model_keys, input_tokens, input_states))
    sampled_toks = jax.vmap(jax.random.categorical)(_gen_keys, generated_outs[:, -1:])
    return sampled_toks, generated_states, gen_keys

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

#### End evolution model implementation


# DEVELOP DATASET

from datasets import load_dataset
dataset_id = "openai/gsm8k" #"AI-MO/NuminaMath-TIR"
train_dataset, test_dataset = load_dataset(dataset_id, "main", split=["train", "test"])
def make_conversation(example):
    # return {"prompt": SYSTEM_PROMPT + "\n\n" + f"User: {example['question']}" + "\n\nAssistant: <think"}
    return {"prompt": f"User: {example['question']}\n\nAssistant: <think"}

train_dataset = train_dataset.map(make_conversation)
test_dataset = test_dataset.map(make_conversation)

def safe_decode(tokens, tokenizer):
    try:
        stop_tokens = np.flatnonzero(tokens==0)
        if stop_tokens.size > 0:
            tokens = tokens[:stop_tokens[0]]
        return legacy_tokenizer.decode(tokens)
    except BaseException as e:
        # print("decoding exception")
        return ""



import re

# code from https://github.com/tianlwang/eval_gsm8k

def extract_predicted_answer(text):
    regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
    regexes_to_ignore =[
        ",",
        "\\$",
        "(?s).*#### ",
        "\\.$"
    ]
    match = re.findall(regex_pattern, text)
    if match:
        match = match[-1]
        if isinstance(match, tuple):
            match = [m for m in match if m][0]
        text = match.strip()

        for regex in regexes_to_ignore:
            text = re.sub(regex, "", text)
        return text
    else:
        print("NO REGEX MATCH FOUND")
        return None

def extract_ground_truth(text):
    return text.split('####')[-1].strip()

def check_accuracy(generated_ans, solution):
    ground_truth_answer = extract_ground_truth(solution)
    # print(f"ground truth answer: {ground_truth_answer}")
    # print("model answer (unparsed)", generated_ans.strip())
    model_answer = extract_predicted_answer(generated_ans.strip())
    # print(f"model answer: {model_answer}; ground truth answer: {ground_truth_answer}")
    return 1.0 if (model_answer == ground_truth_answer) else 0.0
    
def single_fitness(generated_answer, true_answer, i):
    find_idx = generated_answer.find("</think>")
    if find_idx == -1:
        return 0.0
    true_idx = find_idx + len("</think>")
    generated_ans = generated_answer[true_idx:]
    return check_accuracy(generated_answer[true_idx:], true_answer)
    

def batch_fitness(tokens_batch, tokenizer, batch_answers):
    rewards = []
    print("tokens batch shape", tokens_batch.shape)
    tokens_batch = np.array(tokens_batch)
    saw_correct = False
    saw_incorrect = False
    for i, tok_seq in enumerate(tokens_batch):
        gen_ans = safe_decode(tok_seq, tokenizer)
        if len(gen_ans) == 0:
            reward = 0.0
        else:
            reward = single_fitness(gen_ans, batch_answers[i//args.generations_per_prompt], i)
        rewards.append(reward)
        if reward == 0.0 and not saw_incorrect:
            print("Incorrect sample:", i)
            print("*"*20)
            print(gen_ans)
            print("*"*20)
            # print(tok_seq)
            saw_incorrect=True
            
        if reward == 1.0 and not saw_correct:
            print("Correct sample:", i)
            print("*"*20)
            print(gen_ans)
            print("*"*20)
            saw_correct=True
    return jnp.array(rewards)

# DONE DEVELOPING DATASET

def generate_keys_from_master(master_evo_key):
    if args.use_antithetic:
        core_evo_keys = jnp.repeat(jax.random.split(master_evo_key, args.parallel_generations // 2), 2, axis=0)
        evo_keys, gen_keys = jax.vmap(jax.random.split, out_axes=1)(core_evo_keys)
        main_evo_sigma = args.evo_sigma * jnp.ones(args.parallel_generations // 2, dtype=init_state.dtype)
        sigma_antithetic = jnp.stack((main_evo_sigma, -main_evo_sigma), axis=-1).ravel()
    else:
        core_evo_keys = jax.random.split(master_evo_key, args.parallel_generations)
        evo_keys, gen_keys = jax.vmap(jax.random.split, out_axes=1)(core_evo_keys)
        sigma_antithetic = args.evo_sigma * jnp.ones(args.parallel_generations, dtype=init_state.dtype)
    return evo_keys, gen_keys, sigma_antithetic

def _generate_batch(params, model_keys, gen_keys, batch_prompts):
    input_toks = jax.block_until_ready(jnp.zeros((gen_keys.shape[0], 1), dtype=jnp.int32))
    def inner_scan(carry, input_tokens):
        toks, states, gen_keys = carry
        true_input = jnp.where(input_tokens == 0, toks[:, 0], input_tokens)
        toks, states, gen_keys = _forward_and_sample(params, model_keys, true_input, states, gen_keys)
        return (toks, states, gen_keys), true_input
    _, out_tokens = jax.lax.scan(inner_scan, (input_toks, init_states, gen_keys), batch_prompts.T)
    return out_tokens.T

# batch_fitness = reward_functions[args.reward_fn]

def simple_full_update(param, key, scores, lr):
    if args.freeze_nonlora:
        return param
    true_key, sigma = key
    noises = jax.vmap(partial(jax.random.normal, shape=param.shape, dtype=param.dtype))(true_key)
    broadcasted_scores = jnp.reshape(scores, scores.shape + (1,) * len(param.shape))
    broadcasted_sigma = jnp.reshape(sigma, sigma.shape + (1,) * len(param.shape))
    return jnp.astype(param + lr * jnp.mean(broadcasted_scores * noises / broadcasted_sigma, axis=0), param.dtype)

def simple_lora_update(param, key, scores, lr):
    if args.freeze_lora:
        return param
    a, b = param.shape
    true_key, sigma = key
    noises = jax.vmap(partial(jax.random.normal, shape=(a+b, args.lora_dim), dtype=param.dtype))(true_key)
    Bs = noises[:, :b] # Bxbxl
    As = jnp.ones_like(noises[:, b:]) # Bxaxl
    broadcasted_scores = jnp.reshape(scores, scores.shape + (1,) * len(param.shape))
    broadcasted_sigma = jnp.reshape(sigma, sigma.shape + (1,) * len(param.shape)) / lr

    preB = broadcasted_scores / broadcasted_sigma * Bs
    preA = As

    B = jnp.mean(preB, axis=0)
    A = jnp.mean(preA , axis=0)
    actual_grad = A @ B.mT
    return jnp.astype(param + actual_grad, param.dtype)

def _do_update(params, model_keys, raw_scores, lr):

    group_scores = raw_scores.reshape((-1, args.generations_per_prompt))
    true_scores = (group_scores - jnp.mean(group_scores, axis=-1, keepdims=True)).ravel()
    # true_scores = (raw_scores - jnp.mean(raw_scores, keepdims=True)) / jnp.sqrt(jnp.var(raw_scores, keepdims=True) + 1e-5)
    
    def inner_update(param, model_key, lora_map_ans):
        if lora_map_ans == UNCHANGED:
            return param

        update_fn = [simple_full_update, simple_lora_update][lora_map_ans - 1]
        
        if len(model_key[0].shape) == 1:
            return update_fn(param, model_key, true_scores, lr)
        else:
            return jax.lax.scan(lambda _, x: (0, update_fn(x[0], x[1], true_scores, lr)), 0, xs=(param, (jnp.moveaxis(model_key[0], 0, 1), jnp.moveaxis(model_key[1], 0, 1))))[1]

    return jax.tree.map(inner_update, params, model_keys, lora_map)

params = jax.device_put(params, jax.local_devices()[0]) # move it to gpu (or whatever the default device is)
init_state = RWKV.default_state(params, config)
init_states = jnp.repeat(init_state[None], args.parallel_generations, axis=0)


key = jax.random.key(args.seed)
key, master_evo_key = jax.random.split(key)
evo_keys, gen_keys, sigma_antithetic = generate_keys_from_master(master_evo_key)
model_keys = fast_generate_model_keys(params, evo_keys, sigma_antithetic)

print("Compiling generate batch")
start_time = time.time()
generate_batch = jax.jit(_generate_batch).lower(params, model_keys, gen_keys, jnp.zeros((args.parallel_generations, args.generation_length), dtype=jnp.int32)).compile()
print("Compile time", time.time() - start_time)
print("memory info")
print(generate_batch.memory_analysis())
print()
print("Compiling do update")
start_time = time.time()
do_update = jax.jit(_do_update).lower(params, model_keys, jnp.zeros(args.parallel_generations), args.lr).compile()
print("Compile time", time.time() - start_time)
print("memory info")
print(do_update.memory_analysis())

if args.track:
    run = wandb.init(
        project=args.wandb_project,
        config=args,
        name="gsm8k_"+args.wandb_name+f"_lr={args.lr}_sigma={args.evo_sigma}_bs={args.parallel_generations}"
    )
else:
    print("Run name:", "gsm8k_"+args.wandb_name+f"_lr={args.lr}_sigma={args.evo_sigma}_bs={args.parallel_generations}")

max_len = 0

def get_padded_prompt(single_prompt):
    single_prompt = single_prompt[:args.generation_length]
    return single_prompt + [0] * (args.generation_length - len(single_prompt))
    
for epoch in tqdm.trange(args.num_epochs):
    start_time = time.time()
    batch_answers = [train_dataset[(epoch * args.prompts_per_epoch + i) % len(train_dataset)]["answer"] for i in range(args.prompts_per_epoch)]
    batch_prompts = jnp.repeat(jnp.array([get_padded_prompt(tokenizer.encode(train_dataset[(epoch * args.prompts_per_epoch + i) % len(train_dataset)]["prompt"])) for i in range(args.prompts_per_epoch)]), args.generations_per_prompt, axis=0)
    prompt_processing_time = time.time() - start_time

    
    start_time = time.time()
    key, master_evo_key = jax.random.split(key)
    evo_keys, gen_keys, sigma_antithetic = generate_keys_from_master(master_evo_key)
    if epoch == 0:
        print("generating model keys")
    model_keys = fast_generate_model_keys(params, evo_keys, sigma_antithetic)
    key_generation_time = time.time() - start_time

    start_time = time.time()
    if epoch == 0:
        print("generating batch")
    output_batch = jax.block_until_ready(generate_batch(params, model_keys, gen_keys, batch_prompts))
    token_generation_time = time.time() - start_time

    
    start_time = time.time()
    if epoch == 0:
        print("calculating fitness")
    output_scores = jax.block_until_ready(batch_fitness(output_batch, tokenizer, batch_answers))
    fitness_time = time.time() - start_time

    start_time = time.time()
    # print(np.array(output_scores).tolist())
    if epoch == 0:
        print("updating params")
    updated_params = jax.block_until_ready(do_update(params, model_keys, output_scores, args.lr))
    parameter_update_time = time.time() - start_time

    parameter_differences = jax.tree.map(lambda x, y:jnp.mean(jnp.abs(x-y)), params, updated_params)
    lora_updates = jax.tree.reduce(operator.add, jax.tree.map(lambda x, y: x if y == LORA else 0.0, parameter_differences, lora_map)) / jax.tree.reduce(operator.add, jax.tree.map(lambda y: 1.0 if y == LORA else 0.0, lora_map))
    nonlora_updates = jax.tree.reduce(operator.add, jax.tree.map(lambda x, y: x if y == FULL else 0.0, parameter_differences, lora_map)) / jax.tree.reduce(operator.add, jax.tree.map(lambda y: 1.0 if y == FULL else 0.0, lora_map))
    # print(jax.tree.map(lambda x, y:jnp.mean(jnp.abs(x-y)), params, updated_params))
    
    params = updated_params

    stats = {
        "avg_fitness": jnp.mean(output_scores),
        "std_fitness": jnp.std(output_scores),
        "max_fitness": jnp.max(output_scores),
        "min_fitness": jnp.min(output_scores),
        "median_fitness": jnp.median(output_scores),
        "lora_updates": lora_updates,
        "nonlora_updates": nonlora_updates,
        "prompt_preproc_time": prompt_processing_time,
        "keygen_time": key_generation_time,
        "token_gen_time": token_generation_time,
        "fitness_time": fitness_time,
        "update_time": parameter_update_time
    }
    if args.track:
        run.log(stats)
    else:
        print(f"Mean fitness: {jnp.mean(output_scores)}; std fitness: {jnp.std(output_scores)}; max fitness: {jnp.max(output_scores)}; min fitness: {jnp.min(output_scores)}; median fitness: {jnp.median(output_scores)}")
        print("mean parameter diffs")
        print("Lora modules:", lora_updates)
        print("Full modules:", nonlora_updates)
        print("Stats:")
        for k in stats:
            print(f"\t{k}: {stats[k]}")

if args.track:
    run.finish()
