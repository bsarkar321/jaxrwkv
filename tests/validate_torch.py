import os
os.environ['RWKV_JIT_ON'] = '1' if 'RWKV_JIT_ON' not in os.environ else os.environ['RWKV_JIT_ON']
os.environ['RWKV_CUDA_ON'] = '1' if 'RWKV_CUDA_ON' not in os.environ else os.environ['RWKV_CUDA_ON']
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'

import jax
import os
from huggingface_hub.constants import HF_HOME

jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "jaxrwkvcomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

jax.config.update('jax_default_matmul_precision', 'highest')

import torch

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    # turn off TF32 for higher accuracy
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

import jax.numpy as jnp
import numpy as np

from jaxrwkv import get_rand_model, get_model, models

from functools import partial

import tyro

import time

from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class Args:
    model_choice: Literal[tuple(models.keys())] =  "4w0.1B"

    dtype: Optional[str] = "float32"
    rwkv_type: str = "ScanRWKV"
    context: str = "The Eiffel tower is in the city of"

if __name__ == '__main__':
    args = tyro.cli(Args)

    RWKV, params, config, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)

    encoded = tokenizer.encode(args.context)

    use_cuda = torch.cuda.is_available()

    version = args.model_choice[0]

    if version == '7':
        torch_name = models[args.model_choice][2]()[:-4]
        os.environ['RWKV_V7_ON'] = '1'
    else:
        torch_name = models[args.model_choice][2]()

    from rwkv.model import RWKV as RWKV_TORCH


    torch_model = RWKV_TORCH(model=torch_name, strategy=('cuda' if use_cuda else 'cpu') + ' ' + ('bf16' if args.dtype == 'bfloat16' else 'fp32'))

    torch_out, torch_state = torch_model.forward(encoded, None)
    torch_out = torch_out.detach().float().cpu().numpy()
    if torch_state[0].shape == torch_state[1].shape:
        torch_state = np.stack([x.detach().float().cpu().numpy() for x in torch_state])
    else:
        new_torch_state = []
        H = torch_state[1].shape[1]
        for i in range(0, len(torch_state), 3):
            new_torch_state.append(torch_state[i])
            new_torch_state.extend(torch.unbind(torch_state[i+1].reshape((H, -1)), dim=0))
            new_torch_state.append(torch_state[i+2])
        torch_state = np.stack([x.detach().float().cpu().numpy() for x in new_torch_state])

    soft_torch_out = jax.nn.softmax(torch_out)
    t_values, t_indices = jax.lax.top_k(soft_torch_out, 10)
    print("*"*100)
    print("TORCH")
    for i in range(10):
        print(f"{t_values[i].item() * 100}%: {tokenizer.decode([t_indices[i].item()])}")

    init_state = RWKV.default_state(params, config)
    forward_jit = partial(RWKV.forward, config=config)
    out, state = jax.block_until_ready(forward_jit(params, encoded, init_state))
    out = out[len(encoded) - 1]
    soft_out = jax.nn.softmax(out)
    values, indices = jax.lax.top_k(soft_out, 10)
    print("*"*100)
    print("JAX")
    for i in range(10):
        print(f"{values[i].item() * 100}%: {tokenizer.decode([indices[i].item()])}")
    print("*"*100)
    print("TVD=", 0.5*jnp.sum(jnp.abs(soft_torch_out - soft_out)))
    # print(jnp.mean(torch_state, axis=-1))
    compare_state = jnp.roll(state, -1, axis=1)
    # print(jnp.mean(compare_state, axis=-1))
    print("State abs error:", jnp.mean(jnp.abs(jnp.ravel(torch_state) - jnp.ravel(compare_state))))
    print("State rel error:", jnp.mean(jnp.abs((jnp.ravel(torch_state) - jnp.ravel(compare_state))/jnp.ravel(torch_state))))
