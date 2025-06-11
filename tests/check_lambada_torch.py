import os
os.environ['RWKV_JIT_ON'] = '1' if 'RWKV_JIT_ON' not in os.environ else os.environ['RWKV_JIT_ON']
os.environ['RWKV_CUDA_ON'] = '1' if 'RWKV_CUDA_ON' not in os.environ else os.environ['RWKV_CUDA_ON']
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


import jax
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

import torch

@dataclass
class Args:
    model_choice: Literal[tuple(models.keys())] =  "4w0.1B"

    dtype: Optional[str] = "float32"

if __name__ == '__main__':
    args = tyro.cli(Args)

    tokenizer = models[args.model_choice][1]()

    use_cuda = torch.cuda.is_available()
    
    version = args.model_choice[0]

    if version == '7':
        torch_name = models[args.model_choice][2]()[:-4]
        os.environ['RWKV_V7_ON'] = '1'
    else:
        torch_name = models[args.model_choice][2]()

    from rwkv.model import RWKV as RWKV_TORCH


    torch_model = RWKV_TORCH(model=torch_name, strategy=('cuda' if use_cuda else 'cpu') + ' ' + ('bf16' if args.dtype == 'bfloat16' else 'fp32'))
    
    # params = jax.device_put(params, jax.local_devices()[0])
    
    # forward = partial(RWKV.forward, config=config)
    # init_state = RWKV.default_state(params, config)

    # v_forward = jax.jit(jax.vmap(forward, in_axes=(None, 0, None)))

    with open(os.path.join(os.path.dirname(__file__), "lambada_test.jsonl"), "r", encoding="utf-8") as f:
        todo = [json.loads(line) for line in f]
        todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]
        
    compiled_todos = []
    compiled_lens = []
    compiled_lens2 = []
    start_time = time.time()
    seq_len = 256
    print("Preprocessing")
    for d in todo:
        src = [0] + tokenizer.encode(d[0])
        dst = tokenizer.encode(d[1])
        compiled_todos.append(src + dst + [0] * (seq_len - len(src) - len(dst)))
        compiled_lens.append(len(src))
        compiled_lens2.append(len(dst))
    print("Preprocessing time:", time.time() - start_time)

    args.num_sequences = 1

    xsum = 0
    xcnt = 0
    xacc = 0
    start_time = time.time()
    total_runtime = 0
    for iter_num in range(0, len(compiled_todos), args.num_sequences):
        sequences_to_choose = compiled_todos[iter_num:iter_num + args.num_sequences]
        lens_to_choose = compiled_lens[iter_num:iter_num + args.num_sequences]
        lens2_to_choose = compiled_lens2[iter_num:iter_num + args.num_sequences]
        actual_num_seq = len(lens_to_choose)

        if actual_num_seq < args.num_sequences:
            sequences_to_choose += [[0] * seq_len] * (args.num_sequences - actual_num_seq)
            lens_to_choose += [0] * (args.num_sequences - actual_num_seq)

        # print(jnp.array(sequences_to_choose).shape, jnp.array(lens_to_choose).shape)
        # start_time = time.time()
        # out = jax.block_until_ready(v_forward(params, jnp.array(sequences_to_choose), init_state)[0])
        if use_cuda:
            torch.cuda.synchronize()
        mini_start_time = time.time()
        out = torch_model.forward(sequences_to_choose[0], None, True)[0]
        if use_cuda:
            torch.cuda.synchronize()
        mini_end_time = time.time()
        total_runtime += mini_end_time - mini_start_time
        # print(jnp.array(sequences_to_choose)[0])
        # print(time.time() - start_time)
        for t in range(actual_num_seq):
            logits = 0
            correct = True
            for i in range(lens2_to_choose[t]):
                # ooo = jnp.astype(out[t][lens_to_choose[t]-1+i], jnp.float32)
                ooo = jnp.array(out[lens_to_choose[t]-1+i].float().cpu().numpy())
                probs = jax.nn.softmax(ooo, axis=-1)
                logits += jnp.log(probs[sequences_to_choose[t][lens_to_choose[t]+i]])
                if jnp.argmax(probs).item() != sequences_to_choose[t][lens_to_choose[t]+i]:
                    correct = False
                # probs = torch.nn.functional.softmax(ooo, dim=-1)
                # logits += torch.log(probs[sequences_to_choose[t][lens_to_choose[t]+i]])
                # if torch.argmax(probs).item() != sequences_to_choose[t][lens_to_choose[t]+i]:
                #     correct = False
            xcnt += 1
            xsum += logits
            xacc += 1 if correct else 0
            if xcnt % 100 == 0 or xcnt == len(todo):
                time_taken = time.time() - start_time
                start_time = time.time()
                print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc', round(xacc/xcnt*100, 2), 'time', time_taken)
    print(total_runtime, "seconds")

    
