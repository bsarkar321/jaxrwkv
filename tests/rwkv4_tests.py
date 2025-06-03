import jax
import jax.numpy as jnp
import numpy as np

from jaxrwkv import get_model

from functools import partial

import time

RWKV, params, config, tokenizer = get_model("4w0.1B", rwkv_type="AssociativeScanRWKV", verbose=True, dtype=jnp.float32)
params = jax.device_put(params, jax.local_devices()[0]) # move it to gpu (or whatever the default device is)
print(jax.tree.map(lambda x: x.device, params))

init_state = RWKV.default_state(params, config)

context = "The Eiffel tower is in the city of"
answer = " Paris"
encoded = tokenizer.encode(context)
print(context)

forward_jit = partial(RWKV.forward, config=config)

print("RUNNING")
start_time = time.time()
out, state = jax.block_until_ready(forward_jit(params, encoded, init_state, len(encoded)))
end_time = time.time()
print(f"Forward time: {end_time - start_time} seconds")
out = out[len(encoded)-1]
soft_out = jax.nn.softmax(out)
values, indices = jax.lax.top_k(soft_out, 10)
for i in range(10):
    print(f"{values[i].item() * 100}%: {tokenizer.decode([indices[i].item()])}")

old_soft_out = soft_out


print("*"*100)

"""
A1 B2 C3 D4 A5 B6 C7
01 02 03 04 15 06 07

0v Av Bv Cv Dv Av Bv Cv
0  1  1  1  1  1  1  1
-i Ak Bk Ck Dk Ak Bk Ck
0  1  2  3  4  5  6  7


"""



    

encoded_answer = tokenizer.encode(answer)
    
full_context = encoded + encoded_answer + encoded
print(len(full_context))
start_segments = jnp.zeros(len(full_context), dtype=jnp.bool)
start_segments = start_segments.at[len(encoded) + len(encoded_answer)].set(True)
start_time = time.time()
full_out, state = jax.block_until_ready(forward_jit(params, full_context, init_state, len(full_context), start_segments))
end_time = time.time()
print(f"Forward time: {end_time - start_time} seconds")
out = full_out[len(full_context)-1]
soft_out = jax.nn.softmax(out)
values, indices = jax.lax.top_k(soft_out, 10)
for i in range(10):
    print(f"{values[i].item() * 100}%: {tokenizer.decode([indices[i].item()])}")

print()

print("Alt out")
out = full_out[len(encoded)-1]
alt_soft_out = jax.nn.softmax(out)
values, indices = jax.lax.top_k(alt_soft_out, 10)
for i in range(10):
    print(f"{values[i].item() * 100}%: {tokenizer.decode([indices[i].item()])}")

    
print("*"*100)
print("TVD is", 0.5 * jnp.sum(jnp.abs(soft_out - old_soft_out)))



# out, state = jax.block_until_ready(forward_jit(encoded + [0] * (256 - len(encoded)), init_state, params, len(encoded)))
# out = out[len(encoded) - 1]
# for i in range(100):
#     token = sample_logits(np.array(out).astype(np.float64), 1.0, 0.7)
#     try:
#         tmp = tokenizer.decode([token])
#         if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
#             print(tmp, end="", flush=True)
#     except Exception as ex:
#         print("INVALID STRING", ex)

#     out, state = forward_jit([token], state, params, 1)
#     out = out[0]
# print()
