# RWKV in Jax

This repo contains pure jax implementations of RWKV4, RWKV5 (and 5.2), RWKV6, RWKV7, Mamba, and Mamba2. All implementations can be found under the src directory. There are also many convenience features, including:
- Generating final hidden states to enable constant-time generation.
- Support for padding by specifying a "length" in the forward function. (Useful for vmap over sequences of different lengths)
- Resetting in the middle of a sequence, which is helpful for RL training or combining multiple sequences when training.
- Unified interface across all models.
- Directly loading torch models from huggingface and running them in jax.

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
