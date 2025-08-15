import jax
import jax.numpy as jnp

import numpy as np

from functools import partial

def _fastzero(generated_tokens):
    return -jax.numpy.nonzero(generated_tokens == 0, size=1, fill_value=generated_tokens.shape[0]*2)[0][0].astype(jnp.float32)

def _uniquetok(generated_tokens):
    return jnp.sum(jnp.where(jnp.unique_counts(generated_tokens, size=generated_tokens.shape[0]).counts == 0, 0, 1)).astype(jnp.float32)

def _reptok(generated_tokens):
    return jnp.max(jnp.unique_counts(generated_tokens, size=generated_tokens.shape[0]).counts).astype(jnp.float32)

def safe_decode(tokens, tokenizer):
    try:
        return tokenizer.decode(tokens)
    except BaseException as e:
        return ""
    
def digits(batch_generated_tokens, tokenizer):
    stop_tokens = jax.vmap(partial(jnp.flatnonzero, size=1, fill_value=batch_generated_tokens.shape[1]))(batch_generated_tokens == 0)
    numpy_tokens = np.array(batch_generated_tokens)
    num_digits = [sum(c.isdigit() for c in safe_decode(numpy_tokens[i, :stop_tokens[i, 0]], tokenizer)) for i in range(numpy_tokens.shape[0])]
    return jnp.array(num_digits).astype(jnp.float32)

fastzero = jax.jit(jax.vmap(_fastzero))
uniquetok = jax.jit(jax.vmap(_uniquetok))
reptok = jax.jit(jax.vmap(_reptok))

reward_functions = {
    "fastzero": (lambda x, y: fastzero(x)),
    "uniquetok": (lambda x, y: uniquetok(x)),
    "reptok": (lambda x, y: reptok(x)),
    "digits": digits
}
