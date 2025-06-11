import jax
import jax.numpy as jnp

def get_int_component(x):
    parts = x.split('.')
    return tuple([int(p) for p in parts if p.isdigit()])

class LLM:
    @classmethod
    def transform_torch_model(cls, torch_model, dtype=jnp.bfloat16):
        return torch_model

    @classmethod
    def transform_config(cls, config):
        return config
    
    @classmethod
    def load_from_torch(cls, torch_model, config, dtype=jnp.bfloat16):
        """
        Loads from torch (cpu) model and outputs a jax (cpu) model
        """
        w = cls.transform_torch_model(torch_model, dtype=dtype)
        cfg = cls.transform_config(config)
        for k in w.keys():
            w[k] = jnp.array(w[k].float().numpy(), dtype=dtype, device=jax.devices("cpu")[0])

        ans = {}
        ans['blocks'] = {}
        for k in sorted(w.keys(), key=get_int_component):
            parts = k.split('.')
            last = parts.pop()
            here = ans
            add_list = False
            for p in parts:
                if p.isdigit():
                    add_list = True
                else:
                    if p not in here:
                        here[p] = {}
                    here = here[p]
            if not add_list:
                here[last] = w[k]
            else:
                if last not in here:
                    here[last] = [w[k]]
                else:
                    here[last].append(w[k])
        return jax.tree.map(lambda x: jnp.array(x) if isinstance(x, list) else x, ans, is_leaf=lambda x: isinstance(x, list)), cfg

    @classmethod
    def randomize_weights(cls, key, n_layer, n_embd, vocab_size, config, dtype):
        raise NotImplementedError("Randomize Weights is not implemented")

    @classmethod
    def default_state(cls, params, config):
        raise NotImplementedError("Default State is not implemented")

    @classmethod
    def embed(cls, params, config, tokens):
        raise NotImplementedError("Embedding function is not implemented")
    
    @classmethod
    def outhead(cls, params, config, x):
        raise NotImplementedError("Out head is not implemented")

    @classmethod
    def forward_seq(cls, params, config, x, state, length, new_starts):
        raise NotImplementedError("Forward sequence is not implemented")

    @classmethod
    def forward(cls, params, tokens, state, length=None, new_starts=None, config=None):
        """
        Forward pass on a single stream of tokens
        """
        tokens = jnp.array(tokens).ravel()
        if length is None:
            length = jnp.size(tokens)
        if new_starts is None:
            new_starts = jnp.zeros(tokens.shape, dtype=jnp.bool)
        x = cls.embed(params, config, tokens)
        x, state = cls.forward_seq(params, config, x, state, length, new_starts)
        x = cls.outhead(params, config, x)
        return x, state
