# RWKV in Jax

This repo contains pure jax implementations of RWKV4, RWKV5 (and 5.2), RWKV6, RWKV7, Mamba, and Mamba2. All implementations can be found under the src directory. There are also many convenience features, including:
- Generating final hidden states to enable constant-time generation.
- Support for padding by specifying a "length" in the forward function. (Useful for vmap over sequences of different lengths)
- Unified interface across all models.
- Directly loading torch models from huggingface and running them in jax.

## Install Instructions

``` bash
  conda create -n jaxrwkv python=3.10
  conda activate jaxrwkv
  conda install nvidia::cuda-toolkit
  conda install cudnn cuda-version=12 -c nvidia
  pip install -e ".[cuda12,testing]"
```
