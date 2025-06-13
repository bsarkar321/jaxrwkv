# RWKV in Jax

This repo contains pure jax implementations of RWKV4, RWKV5 (and 5.2), RWKV6, and RWKV7 (with Mamba, Mamba2, and SmolLM coming soon). All implementations can be found under the src directory. There are also many convenience features, including:
- Generating final hidden states to enable constant-time generation.
- Support for padding by specifying a "length" in the forward function. (Useful for vmap over sequences of different lengths)
- Resetting in the middle of a sequence, which is helpful for RL training or combining multiple sequences when training.
- Unified interface across all models.
- Directly loading torch models from huggingface and running them in jax.

For nearly all purposes, I'd recommend using the AssociativeScanRWKV implementations. The BaseRWKV is a naive for-loop, which will be slow to compile for long sequences, while the ScanRWKV turns the for-loop into a scan (but is still very slow to compile for training, especially for v7). Use the CudaRWKV implementation at your own risk; these are based on the official cuda kernels from RWKV-LM, but are still highly experimental and do not have all the convenience features at training time.

## Install Instructions

Use pip install to install this package. Additionally, add the optional requirements of:
- [cuda12]: for gpu support
- [macos]: for macos support
- [testing]: for testing against the original implementations (in torch)

``` bash
  conda create -n jaxrwkv python=3.10
  conda activate jaxrwkv
  pip install -e ".[cuda12,testing]"
```

For (experimental) custom cuda kernels, install the [jaxrwkvkernel](https://github.com/bsarkar321/jaxrwkvkernel) package.

## Usage Instructions

You can find a list of supported models in src/jaxrwkv/auto.py as the "models" variable. Names are formatted as (Version)(Tokenizer)(Size); for instance, "4w0.1B" indicates RWKV4 using the World tokenizer with 0.1B parameters. Note that for RWKV7 models, "n" indicates pile models (for gpt "neo" tokenizer), "w" indicates world models, and "g" indicates the g1 reasoning models.

Refer to src/jaxrwkv/llm.py for the general interface. Here is an example usage:

``` python
import jax
import jax.numpy as jnp
import numpy as np

from jaxrwkv import get_model

from functools import partial

import time

RWKV, params, config, tokenizer = get_model("4w0.1B", rwkv_type="AssociativeScanRWKV", verbose=True, dtype=jnp.bfloat16)
params = jax.device_put(params, jax.local_devices()[0]) # move it to gpu (or whatever the default device is)
init_state = RWKV.default_state(params, config)

context = "The Eiffel tower is in the city of"
answer = " Paris"
encoded = tokenizer.encode(context)
print(context)

forward = partial(RWKV.forward, config=config)

start_time = time.time()
out, state = jax.block_until_ready(forward(params, encoded, init_state))
end_time = time.time()
print(f"Forward time: {end_time - start_time} seconds (note: much faster with jax.jit)")
out = out[len(encoded)-1]
soft_out = jax.nn.softmax(out)
values, indices = jax.lax.top_k(soft_out, 10)
for i in range(10):
    print(f"{values[i].item() * 100}%: {tokenizer.decode([indices[i].item()])}")
```

In the forward method, there are also 2 optional parameters:
- length: the length of the sequence to use for generating the final hidden state. When unset, it defaults to the length of the token sequence.
- new_starts: a boolean array of the same shape as tokens. Setting a value to "True" indicates that the token at this position is the start of a new sequence, which means resetting the state at this point. This can be helpful for RL training when handling the ends of episodes or handling multiple separate sequences in language modeling.

## Testing Instructions

The following files within the "tests" directory can help validate the correctness of the implementations. These are very simple tests, so I highly recommend independent verification on your specific domain of interest.

- validate_torch.py: pass in the model_choice and rwkv_type (BaseRWKV, ScanRWKV, AssociativeScanRWKV, CudaRWKV) to check if it is correct. This only tests single sequences (specified by context), and does not check the correctness of "length," "new_starts," or jax vmaps. A low TVD implies that the output distributions are similar for both models, and low state abs and rel errors imply that the generated final states are similar.
- starts_and_length_test.py: similar inputs to validate_torch. This tests whether the length and new_starts parameters are respected by providing the question and answer and resetting before the next question. Both the truncated and full TVD and states should be low, indicating that the resets are working. A major change towards the correct answer (in terms of probabilities) means that information is leaking, implying a bug in the new_starts usage.
- check_lambada.py: pass in the model_choice, rwkv_type, and dtype (float32 or bfloat16) to check the lambada score (ppl and acc). To make it faster (at the cost of vram), you can set num_sequences to the number of sequences you want to process in parallel. Validate these score against "check_lambada_torch.py" which should be similar.
- rand_creation.py: tests whether the get_rand_model creates a valid model for a given version and runs a dummy forward pass. If there are no errors, it is successful.
- rand_grad_test.py: tests whether the gradients are calculated properly, even including non-default init states and random start sequences in the middle. Pass in the version, model_choice, rwkv_type (for the class to test), and validation_rwkv_type (for the implementation with autograd). Typically, this should be used with CudaRWKV since this has a custom backwards function. The initial section of output consists of the mean absolute value of each component of the gradient (first for the model to test and then the "validation" model), and the final section consists of the mean absolute difference. Double check that the errors are reasonably small relative to the gradient magnitudes. NOTE: this does not validate gradients from the output state (i.e. having the loss function contain a state-dependent term) or having gradients flow through into the input state (state tuning).

Example calls for RWKV4:

``` bash
python validate_torch.py --model_choice 4w0.1B --rwkv_type AssociativeScanRWKV --dtype float32
python starts_and_length_test.py --model_choice 4w0.1B --rwkv_type AssociativeScanRWKV --dtype float32 # validation_rwkv_type can be anything that was previously validated by validate_torch
python check_lambada.py --model_choice 4w0.1B --rwkv_type AssociativeScanRWKV --dtype float32
python check_lambada_torch.py --model_choice 4w0.1B --dtype float32
python rand_creation.py --version 4 --n_layer 3 --n_embd 256 --vocab_size 10 --dtype float32 --rwkv_type AssociativeScanRWKV
python rand_grad_test.py --model_choice 4w0.1B --rwkv_type CudaRWKV --batch_size 4 --sequence_length 32 --new_start_prob 0.1 --dtype float32 --validation_rwkv_type ScanRWKV
```

## Implemented Features

|         | BaseRWKV           | ScanRWKV           | AssociativeScanRWKV | CudaRWKV                 |
|---------|--------------------|--------------------|---------------------|--------------------------|
| rwkv4   | :white_check_mark: | :white_check_mark: | :white_check_mark:  | :heavy_exclamation_mark: |
| rwkv5   | :white_check_mark: | :white_check_mark: | :white_check_mark:  | :x:                      |
| rwkv5_2 | :white_check_mark: | :white_check_mark: | :white_check_mark:  | :x:                      |
| rwkv6   | :white_check_mark: | :white_check_mark: | :white_check_mark:  | :x:                      |
| rwkv7   | :white_check_mark: | :white_check_mark: | :white_check_mark:  | :heavy_exclamation_mark: |

:heavy_exclamation_mark: The CudaRWKV implementations for rwkv4 and rwkv7 are incomplete. Notably, neither support gradients from the output state (having the state output as part of the loss function) or into the initial state (state tuning). Additionally, rwkv7 does not support resetting in the middle of sequences due to the "chunked" nature of the cuda kernel. These will fail silently, so double check your outputs against AssociativeScanRWKV before initiating long training sessions.
