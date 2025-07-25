# pip install math_verify
import jax
import os
from huggingface_hub.constants import HF_HOME

jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "jaxrwkvcomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp
import optax

import tyro
from datasets import load_dataset
from dataclasses import dataclass

from jaxrwkv import get_model, models

from typing import Optional, Literal

import time

from functools import partial

import numpy as np

from tqdm import tqdm


def get_input(prompt, completed, sequence_length):
    # outputs padded prompt, cur length, finished
    full_prompt_length = len(prompt)
    cur_length = min(full_prompt_length - completed, sequence_length)
    to_pad = sequence_length - cur_length
    finished = (completed + cur_length == full_prompt_length)
    return prompt[completed:completed+cur_length] + [0] * to_pad, cur_length, finished

def repad_sequences(sequences, sequence_length, batch_size):
    if len(sequences) < batch_size:
        sequences += [([0] * sequence_length, 0, True)] * (batch_size - len(sequences))
    return zip(*sequences)

def preprocess_prompts(RWKV, batch_forward, model, prompts, default_state, sequence_length=64, batch_size=10):
    all_states = jnp.repeat(default_state[None], len(prompts), axis=0)
    prompts_to_handle = np.argsort([len(x) for x in prompts])[::-1].tolist()
    current_length = [0] * len(prompts)
    all_logits = [None] * len(prompts)

    while len(prompts_to_handle) > 0:
        selected_prompt_idxes = prompts_to_handle[:batch_size]
        true_num_prompts_processed = len(selected_prompt_idxes)
        selected_states = jnp.concatenate((all_states[jnp.array(selected_prompt_idxes)], jnp.repeat(default_state[None], batch_size-true_num_prompts_processed, axis=0)))
        padded_prompt, prompt_length, prompt_done = repad_sequences([get_input(prompts[i], current_length[i], sequence_length) for i in selected_prompt_idxes], sequence_length, batch_size)

        # print(selected_prompt_idxes, prompt_length, prompt_done)
        padded_prompt = jnp.array(padded_prompt)
        # print("prompt size", padded_prompt.shape)
        prompt_length = jnp.array(prompt_length)

        # actually get calculations
        generated_outs, generated_states = batch_forward(model, padded_prompt, selected_states, prompt_length)

        all_states = all_states.at[jnp.array(selected_prompt_idxes)].set(generated_states[:true_num_prompts_processed])
        
        for i in range(true_num_prompts_processed):
            idx = selected_prompt_idxes[i]
            current_length[idx] += prompt_length[i].item()
            if prompt_done[i]:
                prompts_to_handle.remove(idx)
                all_logits[idx] = generated_outs[i][prompt_length[i] - 1]
        
    return jnp.array(all_logits), all_states








def generate_batch(key, RWKV, batch_forward, model, states, init_logits, batch_size=100, max_sequence_length=1000):
    print("starting to generate batch", states.shape)
    answers_to_handle = list(range(states.shape[0]))
    answers = [[] for _ in range(states.shape[0])]

    logits = init_logits

    pbar = tqdm(total=(max_sequence_length * len(answers))// batch_size)
    
    while len(answers_to_handle) > 0:
        selected_ans_idxes = answers_to_handle[:batch_size]
        ans_idx_arr = jnp.array(selected_ans_idxes)
        # print("selected idxes", selected_ans_idxes)
        true_num_ans_processed = len(selected_ans_idxes)
        selected_states = jnp.concatenate((states[ans_idx_arr], jnp.repeat(jnp.zeros_like(states[:1]), batch_size-true_num_ans_processed, axis=0)))

        selected_logits = jnp.concatenate((logits[ans_idx_arr], jnp.repeat(jnp.zeros_like(logits[:1]), batch_size-true_num_ans_processed, axis=0)))
        key, _key = jax.random.split(key)
        sampled_tokens = jax.random.categorical(_key, selected_logits)

        # print(sum([len(x) for x in answers]))
        new_logits, new_states = batch_forward(model, sampled_tokens, selected_states, jnp.ones(batch_size, dtype=jnp.int32))

        logits = logits.at[ans_idx_arr].set(new_logits[:true_num_ans_processed, 0])
        states = states.at[ans_idx_arr].set(new_states[:true_num_ans_processed])
        for i in range(true_num_ans_processed):
            idx = selected_ans_idxes[i]
            sampled_token = sampled_tokens[i].item()
            answers[idx].append(sampled_token)
            if sampled_token == 0 or len(answers[idx]) == max_sequence_length:
                answers_to_handle.remove(idx)
        pbar.update(1)
        
    return answers
    
    # out will be list of (list of tokens)


# from math_verify import LatexExtractionConfig, parse, verify


# def accuracy_reward(completions, **kwargs):
#     """Reward function that checks if the completion is the same as the ground truth."""
#     solutions = kwargs["solution"]
#     completion_contents = [completion[0]["content"] for completion in completions]
#     rewards = []
#     for content, solution in zip(completion_contents, solutions):
#         gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
#         answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
#         if len(gold_parsed) != 0:
#             try:
#                 rewards.append(float(verify(answer_parsed, gold_parsed)))
#             except Exception:
#                 rewards.append(0.0)
#         else:
#             rewards.append(1.0)
#     return rewards

# def check_accuracy(solution, content):
#     gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
#     answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
#     if len(gold_parsed) != 0:
#         try:
#             return float(verify(answer_parsed, gold_parsed))
#         except Exception:
#             print("EXCEPTION")
#             return 0.0
#             # rewards.append(0.0)
#     else:
#         print("UNABLE TO PARSE GOLD")
#         return 1.0
#         # rewards.append(1.0)

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
    print(f"model answer: {model_answer}; ground truth answer: {ground_truth_answer}")
    return 1.0 if (model_answer == ground_truth_answer) else 0.0


@dataclass
class Args:
    seed: int = 0
    model_choice: Literal[tuple(models.keys())] =  "7g0.1B"

    rwkv_type: str = "CudaRWKV"
    dtype: Optional[str] = None

    prompts_per_epoch: int = 100

    preprocess_batch_size: int = 10
    preprocess_seq_len: int = 64

    generations_per_prompt: int = 1
    gen_batch_size: int = 10
    max_gen_sequence_length: int = 1000

    rl_batch_size: int = 10
    rl_seq_len: int = 64

    lr: float = 1e-5
    clip_eps: float = 0.2

args = tyro.cli(Args)
assert args.gen_batch_size <= args.prompts_per_epoch * args.generations_per_prompt, f"{args.gen_batch_size} <= {args.prompts_per_epoch} * {args.generations_per_prompt}"

dataset_id = "openai/gsm8k" #"AI-MO/NuminaMath-TIR"
train_dataset, test_dataset = load_dataset(dataset_id, "main", split=["train", "test"])

# SYSTEM_PROMPT = (
#     "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
#     "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
#     "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively"
# )

print(train_dataset)
print(train_dataset[0])

def make_conversation(example):
    # return {"prompt": SYSTEM_PROMPT + "\n\n" + f"User: {example['question']}" + "\n\nAssistant: <think"}
    return {"prompt": f"User: {example['question']}\n\nAssistant: <think"}

train_dataset = train_dataset.map(make_conversation)
test_dataset = test_dataset.map(make_conversation)

print(train_dataset[0])

print(train_dataset[0]["prompt"])


RWKV, params, config, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)
params = jax.device_put(params, jax.local_devices()[0]) # move it to gpu (or whatever the default device is)
# ref_params = jax.tree.map(lambda p: p.copy(), params)
solver = optax.adamw(args.lr)
optimizer = solver.init(jax.tree.map(lambda p: p.copy(), params))
init_state = RWKV.default_state(params, config)
print(jax.tree.map(lambda x:x.shape, params))
print(init_state.device)

forward = jax.jit(partial(RWKV.forward, config=config))
batch_forward = jax.jit(jax.vmap(partial(RWKV.forward, config=config), in_axes=(None, 0, 0, 0))) # params, tokens, state, length
start_time = time.time()

num_epochs = len(train_dataset) // args.prompts_per_epoch

key = jax.random.key(args.seed)

def get_reward(generated_answer, true_answer, i):
    find_idx = ans.find("</think>")
    if find_idx == -1:
        print(i, "unable to find end of think")
        print("*"*20)
        print(ans)
        print("*"*20)
        return 0.0
    true_idx = find_idx + len("</think>")

    generated_ans = ans[true_idx:]
    # content = batch_answers[i//args.generations_per_prompt]
    content = true_answer

    print("="*10, f" {i} ", "="*10)
    print("length=", len(tok_seq))
    # print("_"*20)
    # print(ans)
    # print("$" * 20, " PRED ANSWER ", "$"*20)
    # print(ans[true_idx:])
    # print("_" * 20, " TRUE ANSWER ", "_"*20)
    # print(content)
    print("ACCURACY")
    acc = check_accuracy(generated_ans, content)
    print(i, acc)
    print()
    print()
    return acc

def single_example_loss(params, ref_params, state, ref_state, tokens, answer_tokens, action_mask, advantage):
    # tokens always same shape (for jax jit and DR GRPO)
    T = answer_tokens.shape[0]

    pi, state = forward(params, tokens, state)
    ref_pi, ref_state = forward(ref_params, tokens, ref_state)
    
    pi_logprob = jax.nn.log_softmax(pi)[jnp.arange(T), answer_tokens]
    ref_pi_logprob = jax.nn.log_softmax(ref_pi)[jnp.arange(T), answer_tokens]
    ratio = jnp.exp(pi_logprob - ref_pi_logprob)
    
    token_loss = -jnp.minimum(
        ratio * advantage,
        jnp.clip(ratio, 1-args.clip_eps, 1+args.clip_eps) * advantage
    )
    return jnp.sum(jnp.where(action_mask, token_loss, jnp.zeros_like(token_loss))) / T, (state, ref_state)

def batch_loss(params, ref_params, state, ref_state, tokens, answer_tokens, action_mask, advantage):
    print(state.shape, ref_state.shape, tokens.shape, answer_tokens.shape, action_mask.shape, advantage.shape)
    losses, aux = jax.vmap(single_example_loss, in_axes=(None, None, 0, 0, 0, 0, 0, 0))(params, ref_params, state, ref_state, tokens, answer_tokens, action_mask, advantage)
    return jnp.mean(losses), aux

fast_batch_grad = jax.value_and_grad(batch_loss, has_aux=True)

def do_update(params, optimizer, ref_params, state, ref_state, tokens, answer_tokens, action_mask, advantage):
    (loss, (state, ref_state)), grad = fast_batch_grad(params, ref_params, state, ref_state, tokens, answer_tokens, action_mask, advantage)
    updates, optimizer = solver.update(grad, optimizer, params)
    params = optax.apply_updates(params, updates)
    return params, optimizer, loss, state, ref_state

fast_update_fn = jax.jit(do_update, donate_argnums=(0, 1))

def convert_sample_to_buffer(last_prompt_tok, generation, i):
    over_length = len(generation) % args.rl_seq_len
    padding = np.zeros(args.rl_seq_len-over_length if over_length > 0 else 0, dtype=np.int32)
    full_true_prompt = np.concatenate(([last_prompt_tok], generation, padding))
    assert full_true_prompt.size % args.rl_seq_len == 1
    masks = np.zeros(full_true_prompt.size, dtype=bool)
    masks[1:1+len(generation)] = True
    # consists of last token of prompt + full generation + padding
    return full_true_prompt, masks, full_true_prompt.size // args.rl_seq_len

def get_rl_input(prompt, mask, advantage, iter_num, sequence_length):
    start = iter_num * sequence_length
    return prompt[start: start + sequence_length], prompt[start + 1: start + sequence_length + 1], mask[start + 1: start + sequence_length + 1], advantage

def repad_rl_sequences(sequences, sequence_length, batch_size):
    if len(sequences) < batch_size:
        sequences += [([0] * sequence_length, [0] * sequence_length, [False] * sequence_length, 0.0)] * (batch_size - len(sequences))
    return zip(*sequences)

def rl_process_generations(params, optimizer, update_fn, token_prompts, token_generations, default_state, advantages):
    ref_params = jax.tree.map(lambda p: p.copy(), params)
    full_true_prompts, full_masks, prompt_iterations = list(zip(*[convert_sample_to_buffer(token_prompts[i//args.generations_per_prompt][-1], token_generations[i], i) for i in range(len(token_generations))]))

    _, prompt_states = preprocess_prompts(RWKV, batch_forward, params, token_prompts, default_state, sequence_length=args.preprocess_seq_len, batch_size=args.preprocess_batch_size)
    _, prompt_states_ref = preprocess_prompts(RWKV, batch_forward, ref_params, token_prompts, default_state, sequence_length=args.preprocess_seq_len, batch_size=args.preprocess_batch_size)

    # print("advantages", advantages)
    # print("prompt iterations", prompt_iterations)
    
    generation_states = jnp.repeat(prompt_states, args.generations_per_prompt, axis=0)
    generation_states_ref = jnp.repeat(prompt_states_ref, args.generations_per_prompt, axis=0)

    generations_to_handle = np.argsort(prompt_iterations)[::-1].tolist()
    current_length = [0] * len(token_generations)

    batch_size = args.rl_batch_size

    pbar = tqdm(total=sum(prompt_iterations))

    losses = []
    
    while len(generations_to_handle) > 0:
        selected_generation_idxes = generations_to_handle[:batch_size]
        true_num_generations_processed = len(selected_generation_idxes)
        
        selected_states = jnp.concatenate((generation_states[jnp.array(selected_generation_idxes)], jnp.repeat(default_state[None], batch_size-true_num_generations_processed, axis=0)))
        selected_states_ref = jnp.concatenate((generation_states_ref[jnp.array(selected_generation_idxes)], jnp.repeat(default_state[None], batch_size-true_num_generations_processed, axis=0)))

        rl_tokens, rl_answer_tokens, rl_action_mask, rl_advantages = repad_rl_sequences([get_rl_input(full_true_prompts[i], full_masks[i], advantages[i], current_length[i], args.rl_seq_len) for i in selected_generation_idxes], args.rl_seq_len, batch_size)

        params, optimizer, loss, new_selected_states, new_selected_states_ref = update_fn(params, optimizer, ref_params, selected_states, selected_states_ref, jnp.array(rl_tokens), jnp.array(rl_answer_tokens), jnp.array(rl_action_mask), jnp.array(rl_advantages))
        # print("loss=", loss)
        losses.append(loss)

        generation_states = generation_states.at[jnp.array(selected_generation_idxes)].set(new_selected_states[:true_num_generations_processed])
        generation_states_ref = generation_states_ref.at[jnp.array(selected_generation_idxes)].set(new_selected_states_ref[:true_num_generations_processed])

        for i in range(true_num_generations_processed):
            idx = selected_generation_idxes[i]
            current_length[idx] += 1
            if current_length[idx] == prompt_iterations[idx]:
                generations_to_handle.remove(idx)

        pbar.update(true_num_generations_processed)

    print("losses", losses)

    return params, optimizer


for t in range(num_epochs):
    print(f"epoch {t}/{num_epochs}")

    batch_answers = [train_dataset[t * args.prompts_per_epoch + i]["answer"] for i in range(args.prompts_per_epoch)]

    batch_prompts = [tokenizer.encode(train_dataset[t * args.prompts_per_epoch + i]["prompt"]) for i in range(args.prompts_per_epoch)]
    num_tokens = sum(len(x) for x in batch_prompts)

    start_time = time.time()
    logits, states = preprocess_prompts(RWKV, batch_forward, params, batch_prompts, init_state, sequence_length=args.preprocess_seq_len, batch_size=args.preprocess_batch_size)
    end_time = time.time()
    print("preprocess time is", end_time-start_time, "which is", num_tokens/(end_time-start_time), "tok/s for", num_tokens, "tokens")
    
    # key, _key = jax.random.split(key)
    # print(jax.random.categorical(_key, logits))

    gen_logits = jnp.repeat(logits, args.generations_per_prompt, axis=0)
    gen_states = jnp.repeat(states, args.generations_per_prompt, axis=0)
    start_time = time.time()
    key, _key = jax.random.split(key)
    generated_tokens = generate_batch(_key, RWKV, batch_forward, params, gen_states, gen_logits, batch_size=args.gen_batch_size, max_sequence_length=args.max_gen_sequence_length)
    end_time = time.time()
    total_generated_tokens = sum(len(x) for x in generated_tokens)
    print("generation time is", end_time-start_time, "which is", total_generated_tokens/(end_time-start_time), "tok/s for", total_generated_tokens, "tokens")

    rewards = []
    
    for i, tok_seq in enumerate(generated_tokens):
        ans = tokenizer.decode(tok_seq)
        reward = get_reward(ans, batch_answers[i//args.generations_per_prompt], i)
            
        rewards.append(reward)

    np_rewards = np.array(rewards).reshape((-1, args.generations_per_prompt))
    np_advantages = (np_rewards - np.mean(np_rewards, axis=-1, keepdims=True)).flatten()
    print("ACCURACY:", np.mean(np_rewards))
    print("LENGTHS:", [len(x) for x in generated_tokens])

    params, optimizer = rl_process_generations(params, optimizer, fast_update_fn, batch_prompts, generated_tokens, init_state, np_advantages)

# print("Starting preprocess")
# out, state = jax.block_until_ready(forward(params, encoded, init_state))
# out = out[-1]
# end_time = time.time()
# print("Done preprocess", end_time - start_time, " which is ", len(encoded)/(end_time-start_time), "tok/s")

# key = jax.random.key(args.seed)

# start_time = time.time()

# generated_tokens = []

# while True:
#     key, _key = jax.random.split(key)
#     sampled_token = jax.random.categorical(_key, out).item()
#     generated_tokens.append(sampled_token)
#     if sampled_token == 0:
#         break
#     print(tokenizer.decode([sampled_token]), end="", flush=True)
#     out, state = forward(params, [sampled_token], state)

# end_time = time.time()
# print("time taken", end_time - start_time, " which is ", len(generated_tokens)/(end_time-start_time), "tok/s")
# print(generated_tokens)

