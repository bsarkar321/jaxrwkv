import jax
import jax.numpy as jnp

from functools import partial

from .llm import LLM

def layer_norm(x, w, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    std = jnp.sqrt(var + eps)
    return (x - mean) / std * w['weight'] + w['bias']

def group_norm(x, num_groups, weight, bias, eps):
    N, C = x.shape[:2]
    G = num_groups
    x = x.reshape(N, G, C // G, *x.shape[2:])
    mean = jnp.mean(x, axis=(2, *range(3, x.ndim)), keepdims=True)
    var = jnp.var(x, axis=(2, *range(3, x.ndim)), keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps)
    x = x.reshape(N, C, *x.shape[3:])
    gamma = weight.reshape(1, C, *([1] * (x.ndim - 2)))
    beta = bias.reshape(1, C, *([1] * (x.ndim - 2)))
    return gamma * x + beta

def orthogonal_(key, rows, cols, gain, dtype):
    flattened = jax.random.normal(key=key, shape=(rows, cols))
    if rows < cols:
        flattened = flattened.T

    q, r = jnp.linalg.qr(flattened)
    d = jnp.diag(r, 0)
    ph = jnp.sign(d)
    q *= ph * gain

    if rows < cols:
        q = q.T
    return q.astype(dtype)

def inner_orthogonal(flattened, rows, cols, gain):
    if rows < cols:
        flattened = flattened.T
    q, r = jnp.linalg.qr(flattened)
    d = jnp.diag(r, 0)
    ph = jnp.sign(d)
    q *= ph * gain

    if rows < cols:
        q = q.T
    return q

def p_orthogonal_(key, blocks, rows, cols, gain, dtype):
    flattened = jax.random.normal(key=key, shape=(blocks, rows, cols))

    return jax.vmap(inner_orthogonal, in_axes=(0, None, None, None))(flattened, rows, cols, gain).astype(dtype)


class BaseRWKV(LLM):
    @classmethod
    def transform_torch_model(cls, torch_model, dtype=jnp.bfloat16):
        w = torch_model
        w['ln0.weight'] = w['blocks.0.ln0.weight']
        w['ln0.bias'] = w['blocks.0.ln0.bias']
        del w['blocks.0.ln0.weight']
        del w['blocks.0.ln0.bias']
        for k in w.keys():
            if '.time_' in k:
                w[k] = w[k].squeeze()
            if '.time_first' in k:
                w[k] = w[k].unsqueeze(-1)
        return w

    @classmethod
    def randomize_att(cls, att_key, n_layer, n_embd, vocab_size, dtype):
        ratio_0_to_1 = jnp.arange(n_layer, dtype=dtype) / (n_layer - 1)
        ratio_1_to_almost_0 = 1.0 - jnp.arange(n_layer, dtype=dtype) / (n_layer)
        
        params = {}
        h = jnp.arange(n_embd, dtype=dtype) / (n_embd - 1)
        params['time_decay'] = jnp.reshape(jnp.astype(-6 + 5 * jnp.pow(h[None], 0.7 + 1.3 * ratio_0_to_1[:, None]), dtype), (n_layer, -1, 64))[:, :, 0]
        zigzag = ((jnp.arange(n_embd) + 1) % 3 - 1) * 0.1
        params['time_first'] = jnp.reshape(jnp.astype(ratio_0_to_1[:, None] * (1-h) + zigzag[None], dtype), (n_layer, -1, 64))[:, :, :1]

        x = jnp.arange(n_embd, dtype=dtype) / n_embd
        params['time_mix_k'] = jnp.pow(x[None], ratio_1_to_almost_0[:, None])
        params['time_mix_v'] = (jnp.pow(x[None], ratio_1_to_almost_0[:, None]) + 0.3 * ratio_0_to_1[:, None])
        params['time_mix_r'] = jnp.pow(x[None], 0.5 * ratio_1_to_almost_0[:, None])


        orthos = jnp.astype(jax.random.orthogonal(att_key, n_embd, shape=(4, n_layer)), dtype)
        
        # key, output, receptance, value
        params['key'] = {'weight': orthos[0] * 0.1}
        params['output'] = {'weight': jnp.zeros((n_layer, n_embd, n_embd), dtype=dtype)}
        params['receptance'] = {'weight': orthos[1]}
        params['value'] = {'weight': orthos[2]}
        params['ln_x'] = {'bias': jnp.zeros((n_layer, n_embd), dtype=dtype), 'weight': jnp.astype(jnp.repeat(((1 + jnp.arange(n_layer)) / n_layer)[:, None] ** 0.7, n_embd, axis=1), dtype)}
        
        return params

    @classmethod
    def randomize_ffn(cls, ffn_key, n_layer, n_embd, vocab_size, dtype):
        ratio_1_to_almost_0 = 1.0 - jnp.arange(n_layer, dtype=dtype) / (n_layer)
        params = {}
        x = jnp.arange(n_embd, dtype=dtype) / n_embd
        params['time_mix_k'] = jnp.pow(x[None], ratio_1_to_almost_0[:, None])
        params['time_mix_r'] = jnp.pow(x[None], ratio_1_to_almost_0[:, None])

        hidden_sz = int(3.5 * n_embd)
        params['key'] = {'weight': p_orthogonal_(ffn_key, n_layer, hidden_sz, n_embd, 1.0, dtype)}
        params['receptance'] = {'weight': jnp.zeros((n_layer, n_embd, n_embd), dtype=dtype)}
        params['value'] = {'weight': jnp.zeros((n_layer, n_embd, hidden_sz), dtype=dtype)}

        return params

    @classmethod
    def randomize_weights(cls, key, n_layer, n_embd, vocab_size, config, dtype):
        att_key, ffn_key, emb_key, head_key = jax.random.split(key, 4)
        emb_scale = 1e-4
        head_scale = 0.5 if vocab_size <= n_embd else 0.5 * jnp.sqrt(vocab_size/n_embd)
        params = {
            'blocks': {
                'att': cls.randomize_att(att_key, n_layer, n_embd, vocab_size, dtype),
                'ffn': cls.randomize_ffn(ffn_key, n_layer, n_embd, vocab_size, dtype),
                'ln1': {'bias': jnp.zeros((n_layer, n_embd), dtype=dtype), 'weight': jnp.ones((n_layer, n_embd), dtype=dtype)},
                'ln2': {'bias': jnp.zeros((n_layer, n_embd), dtype=dtype), 'weight': jnp.ones((n_layer, n_embd), dtype=dtype)},
            },
            'emb': {'weight': jax.random.uniform(key=emb_key, shape=(vocab_size, n_embd), minval=-emb_scale, maxval=emb_scale, dtype=dtype)},
            'head': {'weight': orthogonal_(head_key, vocab_size, n_embd, head_scale, dtype)},
            'ln0': {'bias': jnp.zeros(n_embd, dtype=dtype), 'weight': jnp.ones(n_embd, dtype=dtype)},
            'ln_out': {'bias': jnp.zeros(n_embd, dtype=dtype), 'weight': jnp.ones(n_embd, dtype=dtype)},
        }
        return params, config

    @classmethod
    def default_state(cls, params, config):
        n_embd = params['emb']['weight'].shape[1]
        n_layer = params['blocks']['att']['time_decay'].shape[0]
        n_head = params['blocks']['att']['time_decay'][0].shape[0]
        head_size = params['blocks']['ln1']['weight'][0].shape[0] // n_head
        return jnp.zeros((n_layer, (2 + head_size), n_embd), dtype=params['emb']['weight'].dtype)

    @classmethod
    def embed(cls, params, config, tokens):
        return params['emb']['weight'][tokens]
    
    @classmethod
    def outhead(cls, params, config, x):
        return layer_norm(x, params['ln_out']) @ params['head']['weight'].T

    @classmethod
    def channel_mixing_seq(cls, x, state, ffn, length, new_starts):
        sx = jnp.concatenate([state, x[:-1, :]], dtype=x.dtype)
        sx = jnp.where(new_starts[:, None], jnp.zeros_like(sx), sx)
        xk = x * ffn['time_mix_k'] + sx * (1 - ffn['time_mix_k'])
        xr = x * ffn['time_mix_r'] + sx * (1 - ffn['time_mix_r'])
        r = jax.nn.sigmoid(xr @ ffn['receptance']['weight'].T)
        k = jnp.square(jax.nn.relu(xk @ ffn['key']['weight'].T))
        return r * (k @ ffn['value']['weight'].T), x[length - 1]

    @classmethod
    def inner_loop(cls, r, k, v, w, time_first, s, length, new_starts):
        out = jnp.empty_like(r)
        out_s = s

        reset_s = jnp.zeros_like(s)
        for t in range(r.shape[0]):
            s = jax.lax.select(new_starts[t], reset_s, s)
            
            rt = jnp.expand_dims(r[t], 1)
            kt = jnp.expand_dims(k[t], 2)
            vt = jnp.expand_dims(v[t], 1)
            at = kt@vt
            out = out.at[t].set((rt @ (time_first * at + s)).squeeze(1))
            s = jnp.astype(at + w * s, r.dtype)
            out_s = jax.lax.select(t < length, s, out_s)
        return out_s, out

    @classmethod
    def time_mixing_seq(cls, x, state, att, length, new_starts, H, S):
        T, C = x.shape

        sx = jnp.concatenate([state[:1], x[:-1, :]], dtype=x.dtype)
        sx = jnp.where(new_starts[:, None], jnp.zeros_like(sx), sx)
        xk = x * att['time_mix_k'] + sx * (1 - att['time_mix_k'])
        xv = x * att['time_mix_v'] + sx * (1 - att['time_mix_v'])
        xr = x * att['time_mix_r'] + sx * (1 - att['time_mix_r'])
        state = state.at[0].set(x[length-1])

        r = jnp.reshape(xr @ att['receptance']['weight'].T, (T, H, S))
        k = jnp.reshape(xk @ att['key']['weight'].T, (T, H, S))
        v = jnp.reshape(xv @ att['value']['weight'].T, (T, H, S))

        w = jnp.reshape(jnp.exp(-jnp.exp(att['time_decay'])), (-1, 1, 1))
        u = jnp.reshape(jnp.exp(att['time_first']), (-1, 1, 1))

        s = jnp.reshape(state[1:, :],(H, S, S))

        state_new, out = cls.inner_loop(r, k, v, w, u, s, length, new_starts)
        state = state.at[1:].set(state_new.reshape(S, -1))

        x = out.reshape(T, H*S)

        x = group_norm(x, num_groups=H, weight=att['ln_x']['weight'], bias=att['ln_x']['bias'], eps = 64e-5)
        return x @ att['output']['weight'].T, state

    @classmethod
    def forward_seq(cls, params, config, x, state, length, new_starts):
        n_layer = params['blocks']['att']['time_decay'].shape[0]
        n_head = params['blocks']['att']['time_decay'][0].shape[0]
        head_size = params['blocks']['ln1']['weight'][0].shape[0] // n_head
        x = layer_norm(x, params['ln0'])

        @partial(jax.checkpoint,
                 policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        def block_loop(x, inputs):
            block, state = inputs
            x_new, s = cls.time_mixing_seq(layer_norm(x, block['ln1']), state[1:], block['att'], length, new_starts, n_head, head_size)
            state = state.at[1:].set(s)
            x = x + x_new
            
            x_new, s = cls.channel_mixing_seq(layer_norm(x, block['ln2']), state[:1], block['ffn'], length, new_starts)
            state = state.at[0].set(s)
            x = x + x_new
            return x, state

        x, state = jax.lax.scan(block_loop, x, (params['blocks'], state))
        return x, state


class ScanRWKV(BaseRWKV):
    @classmethod
    def inner_loop(cls, r, k, v, w, time_first, s, length, new_starts):
        idxes = jnp.arange(jnp.size(r, axis=0))
        reset_s = jnp.zeros_like(s)
        def scan_loop(inner_states, x):
            out_s, inner_state = inner_states
            r_t, k_t, v_t, t = x
            inner_state = jax.lax.select(new_starts[t], reset_s, inner_state)
            rt = jnp.expand_dims(r_t, 1)
            kt = jnp.expand_dims(k_t, 2)
            vt = jnp.expand_dims(v_t, 1)
            at = kt@vt
            out_t = jnp.astype((rt @ (time_first * at + inner_state)).squeeze(1), r_t.dtype) # set dtype here
            state = jnp.astype(at + w * inner_state, r_t.dtype)
            out_s = jax.lax.select(t < length, state, out_s)
            return (out_s, state), out_t
        (s, _), out = jax.lax.scan(scan_loop, (s, s), (r, k, v, idxes), unroll=64)
        return jnp.astype(s, r.dtype), jnp.astype(out, r.dtype)

class AssociativeScanRWKV(BaseRWKV):
    @classmethod
    def inner_loop(cls, r, k, v, w, time_first, s, length, new_starts):
        R = jnp.expand_dims(r, 2)
        K = jnp.expand_dims(k, 3)
        V = jnp.expand_dims(v, 2)

        A = K @ V
        all_A = jnp.concatenate((s[None], A), dtype=r.dtype)
        w = jnp.repeat(jnp.expand_dims(w, 0), A.shape[0], axis=0)
        all_W = jnp.concatenate((jnp.zeros_like(w[0:1]), w))
        new_starts = jnp.concatenate((new_starts, jnp.zeros_like(new_starts[:1])))[:, None, None, None]

        all_A = jnp.where(new_starts, jnp.zeros_like(all_A), all_A)
        all_W = jnp.where(new_starts, jnp.zeros_like(all_W), all_W)
        def a_scan_loop(elem1, elem2):
            A1, W1, r1 = elem1
            A2, W2, r2 = elem2

            A12 = A2 + W2 * A1
            W12 = W2 * W1
            return jnp.where(r2, A2, A12), jnp.where(r2, W2, W12), jnp.where(r2, r2, r1)
        scan_s, scan_w, _ = jax.lax.associative_scan(a_scan_loop, (all_A, all_W, new_starts))

        out = jnp.astype((R @ (time_first * A + scan_s[:-1])).squeeze(2), r.dtype)
        s = scan_s[length]
        return jnp.astype(s, r.dtype), out
