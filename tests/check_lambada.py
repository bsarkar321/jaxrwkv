import jax
import os
from huggingface_hub.constants import HF_HOME

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

    num_sequences: int = 1

if __name__ == '__main__':
    args = tyro.cli(Args)

    RWKV, params, config, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)
    params = jax.device_put(params, jax.local_devices()[0])
    
    forward = partial(RWKV.forward, config=config)
    init_state = RWKV.default_state(params, config)

    v_forward = jax.jit(jax.vmap(forward, in_axes=(None, 0, None)))

    seq_len = 256

    with open(os.path.join(os.path.dirname(__file__), "lambada_test.jsonl"), "r", encoding="utf-8") as f:
        todo = [json.loads(line) for line in f]
        todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]


    start_time = time.time()
    print("compiling")
    jax.block_until_ready(v_forward(params, jnp.array([[0] * seq_len] * args.num_sequences), init_state)[0])
    print("Compile time:", time.time() - start_time)
        
    compiled_todos = []
    compiled_lens = []
    compiled_lens2 = []
    start_time = time.time()
    print("Preprocessing")
    for d in todo:
        src = [0] + tokenizer.encode(d[0])
        dst = tokenizer.encode(d[1])
        compiled_todos.append(src + dst + [0] * (seq_len - len(src) - len(dst)))
        compiled_lens.append(len(src))
        compiled_lens2.append(len(dst))
    print("Preprocessing time:", time.time() - start_time)

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
        mini_start_time = time.time()
        out = jax.block_until_ready(v_forward(params, jnp.array(sequences_to_choose), init_state)[0])
        mini_end_time = time.time()
        total_runtime += mini_end_time - mini_start_time
        # print(jnp.array(sequences_to_choose)[0])
        # print(time.time() - start_time)
        for t in range(actual_num_seq):
            logits = 0
            correct = True
            for i in range(lens2_to_choose[t]):
                ooo = jnp.astype(out[t][lens_to_choose[t]-1+i], jnp.float32)
                probs = jax.nn.softmax(ooo, axis=-1)
                logits += jnp.log(probs[sequences_to_choose[t][lens_to_choose[t]+i]])
                if jnp.argmax(probs).item() != sequences_to_choose[t][lens_to_choose[t]+i]:
                    correct = False
            xcnt += 1
            xsum += logits
            xacc += 1 if correct else 0
            if xcnt % 100 == 0 or xcnt == len(todo):
                time_taken = time.time() - start_time
                start_time = time.time()
                print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc', round(xacc/xcnt*100, 2), 'time', time_taken)

    print(total_runtime, "seconds")

    
