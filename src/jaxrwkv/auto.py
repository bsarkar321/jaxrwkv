from .tokenizer import GptTokenizer, WorldTokenizer

from . import rwkv4, rwkv5, rwkv5_2

from huggingface_hub.constants import HF_HOME
from huggingface_hub import hf_hub_download

from pathlib import Path

import pickle

import jax
import jax.numpy as jnp

suffix = ".model"

versions = {
    "4": rwkv4,
    "5": rwkv5,
    "5_2": rwkv5_2,
}

models = {
    "4w0.1B": (rwkv4, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-4-world", filename="RWKV-4-World-0.1B-v1-20230520-ctx4096.pth")), None),
    "4w0.4B": (rwkv4, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-4-world", filename="RWKV-4-World-0.4B-v1-20230529-ctx4096.pth")), None),
    "4w1.5B": (rwkv4, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-4-world", filename="RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth")), None),
    "4w3B": (rwkv4, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-4-world", filename="RWKV-4-World-3B-v1-20230619-ctx4096.pth")), None),
    "4w7B": (rwkv4, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-4-world", filename="RWKV-4-World-7B-v1-20230626-ctx4096.pth")), None),

    "5w0.1B": (rwkv5, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-5-world", filename="RWKV-5-World-0.1B-v1-20230803-ctx4096.pth")), None),
    "5w0.4B": (rwkv5_2, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-5-world", filename="RWKV-5-World-0.4B-v2-20231113-ctx4096.pth")), None),
    "5w1.5B": (rwkv5_2, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-5-world", filename="RWKV-5-World-1B5-v2-20231025-ctx4096.pth")), None),
    "5w3B": (rwkv5_2, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-5-world", filename="RWKV-5-World-3B-v2-20231113-ctx4096.pth")), None),
    "5w7B": (rwkv5_2, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-5-world", filename="RWKV-5-World-7B-v2-20240128-ctx4096.pth")), None),

}

def get_rand_model(seed, version, n_layer, n_embd, vocab_size, config=None, dtype=None, rwkv_type="ScanRWKV", verbose=False):
    rwkv = versions[version]
    RWKV = getattr(rwkv, rwkv_type)
    if dtype is None:
        dtype = jnp.float32 if version.startswith('m') else jnp.bfloat16
    elif isinstance(dtype, str):
        dtype = jnp.bfloat16 if dtype == 'bfloat16' else jnp.float32
    if verbose:
        print(dtype)
    
    model_name = f"{seed}_{version}_{n_layer}_{n_embd}_{vocab_size}"

    path = Path(HF_HOME, "jaxrwkv_cache", f"{model_name}_{str(dtype.dtype)}.model")
    if path.is_file():
        if verbose:
            print("loading from", path)
        rwkv_params, config = load(path)
    else:
        key = jax.random.key(seed)
        with jax.default_device(jax.devices("cpu")[0]):
            rwkv_params, config = RWKV.randomize_weights(key, n_layer, n_embd, vocab_size, config, dtype)
        if verbose:
            print("saving to", path)
        save((rwkv_params, config), path)

    return RWKV, rwkv_params, config

def get_model(model_name, dtype=None, rwkv_type="ScanRWKV", verbose=False):
    rwkv, tok_cls, model_name_fn, config_fn = models[model_name]
    RWKV = getattr(rwkv, rwkv_type)
    rwkv_tokenizer = tok_cls()

    if dtype is None:
        dtype = jnp.float32 if model_name.startswith('m') else jnp.bfloat16
    elif isinstance(dtype, str):
        dtype = jnp.bfloat16 if dtype == 'bfloat16' else jnp.float32
    if verbose:
        print(dtype)

    path = Path(HF_HOME, "jaxrwkv_cache", f"{model_name}_{str(dtype.dtype)}.model")
    if path.is_file():
        if verbose:
            print("loading from", path)
        rwkv_params, config = load(path)
    else:
        import torch
        MODEL_NAME = model_name_fn()
        torch_model = torch.load(MODEL_NAME, map_location='cpu', weights_only=True)
        config = config_fn() if config_fn is not None else None
        rwkv_params, config = RWKV.load_from_torch(torch_model, config, dtype=dtype)
        if verbose:
            print("saving to", path)
        save((rwkv_params, config), path, True)
    return RWKV, rwkv_params, config, rwkv_tokenizer

def get_tokenizer(tokenizer_name):
    if tokenizer_name == "WorldTokenizer":
        return WorldTokenizer
    if tokenizer_name == "GptTokenizer":
        return GptTokenizer
    raise NotImplementedError(f"No such tokenizer {tokenizer_name}")
    

def save(model: any, path: str | Path, overwrite: bool = False):
    """
    Save the Any model as a file given a path.

    See https://github.com/google/jax/issues/2116#issuecomment-580322624

    :param model: The Any model you want to save
    :param path: The path to save the model to
    :param overwrite: Set to true to allow overwriting over existing file
    """
    path = Path(path)
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(model, file)

def load(path: str | Path) -> any:
    """
    Read the Any model from a file

    See https://github.com/google/jax/issues/2116#issuecomment-580322624

    :param path: The path to read the model from
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix != suffix:
        raise ValueError(f'Not a {suffix} file: {path}')
    with jax.default_device(jax.devices("cpu")[0]):
        with open(path, 'rb') as file:
            data = pickle.load(file)
    return data
