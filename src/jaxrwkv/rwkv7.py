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
        import torch
        w = torch_model
        w['ln0.weight'] = w['blocks.0.ln0.weight']
        w['ln0.bias'] = w['blocks.0.ln0.bias']
        w['blocks.0.att.v0'] = torch.zeros_like(w['blocks.1.att.v0'])
        w['blocks.0.att.v1'] = torch.zeros_like(w['blocks.1.att.v1'])
        w['blocks.0.att.v2'] = torch.zeros_like(w['blocks.1.att.v2'])
        del w['blocks.0.ln0.weight']
        del w['blocks.0.ln0.bias']
        for k in w.keys():
            if '.x_' in k or '.k_' in k or '.a0' in k or '.v0' in k or '.w0' in k:
                w[k] = w[k].squeeze()
        return w

    @classmethod
    def randomize_att(cls, att_key, n_layer, n_embd, vocab_size, dtype):
        ratio_0_to_1 = jnp.arange(n_layer, dtype=dtype) / (n_layer - 1)
        ratio_1_to_almost_0 = 1.0 - jnp.arange(n_layer, dtype=dtype) / (n_layer)
        ddd = jnp.arange(n_embd, dtype=dtype) / n_embd
        params = {}

        params['x_r'] = 1 - jnp.pow(ddd[None], 0.2 * ratio_1_to_almost_0[:, None])
        params['x_w'] = 1 - jnp.pow(ddd[None], 0.9 * ratio_1_to_almost_0[:, None])
        params['x_k'] = 1 - jnp.pow(ddd[None], 0.7 * ratio_1_to_almost_0[:, None])
        params['x_v'] = 1 - jnp.pow(ddd[None], 0.7 * ratio_1_to_almost_0[:, None])
        params['x_a'] = 1 - jnp.pow(ddd[None], 0.9 * ratio_1_to_almost_0[:, None])
        params['x_g'] = 1 - jnp.pow(ddd[None], 0.2 * ratio_1_to_almost_0[:, None])

        N = 64
        H = n_embd // 64

        linear = jnp.arange(n_embd, dtype=dtype) / (n_embd - 1) - 0.5
        zigzag0 = ((jnp.arange(n_embd) % N) - ((N - 1) / 2)) / ((N - 1) / 2)
        zigzag = jnp.astype(zigzag0 * jnp.abs(zigzag0), dtype)
        
        www = -6 + 6 * jnp.pow(jnp.arange(n_embd, dtype=dtype)[None] / (n_embd - 1), 1 + 1 * ratio_0_to_1[:, None] ** 0.3)
        linear = jnp.repeat(linear[None], n_layer, axis=0)
        zigzag = jnp.repeat(zigzag[None], n_layer, axis=0)

        D_DECAY_LORA = max(32, int(round(  (1.8*(n_embd**0.5))  /32)*32))
        att_key, _key = jax.random.split(att_key)
        params['w0'] = www + 0.5 + zigzag * 2.5
        params['w1'] = jnp.zeros((n_layer, n_embd, D_DECAY_LORA), dtype=dtype)
        params['w2'] = p_orthogonal_(_key, n_layer, D_DECAY_LORA, n_embd, 0.1, dtype)

        D_AAA_LORA = max(32, int(round(  (1.8*(n_embd**0.5))  /32)*32))
        att_key, _key = jax.random.split(att_key)
        params['a0'] = -0.19 + zigzag * 0.3 + linear * 0.4
        params['a1'] = jnp.zeros((n_layer, n_embd, D_AAA_LORA), dtype=dtype)
        params['a2'] = p_orthogonal_(_key, n_layer, D_AAA_LORA, n_embd, 0.1, dtype)

        
        D_MV_LORA = max(32, int(round(  (1.3*(n_embd**0.5))  /32)*32))
        att_key, _key = jax.random.split(att_key)
        params['v0'] = 0.73 - linear * 0.4
        params['v1'] = jnp.zeros((n_layer, n_embd, D_MV_LORA), dtype=dtype)
        params['v2'] = p_orthogonal_(_key, n_layer, D_MV_LORA, n_embd, 0.1, dtype)


        D_GATE_LORA = max(32, int(round(  (0.6*(n_embd**0.8))  /32)*32))
        att_key, _key = jax.random.split(att_key)
        params['g1'] = jnp.zeros((n_layer, n_embd, D_GATE_LORA), dtype=dtype)
        params['g2'] = p_orthogonal_(_key, n_layer, D_GATE_LORA, n_embd, 0.1, dtype)


        params['k_k'] = 0.71 - linear * 0.1
        params['k_a'] = 1.02 + jnp.zeros_like(linear)
        params['r_k'] = jnp.zeros((n_layer, H, N), dtype=dtype) - 0.04

        orthos = jnp.astype(jax.random.orthogonal(att_key, n_embd, shape=(4, n_layer)), dtype)        
        params['receptance'] = {'weight': orthos[1]}
        params['key'] = {'weight': orthos[0] * 0.1}
        params['value'] = {'weight': orthos[2]}
        params['output'] = {'weight': jnp.zeros((n_layer, n_embd, n_embd), dtype=dtype)}
        params['ln_x'] = {'bias': jnp.zeros((n_layer, n_embd), dtype=dtype), 'weight': jnp.astype(jnp.repeat(((1 + jnp.arange(n_layer)) / n_layer)[:, None] ** 0.7, n_embd, axis=1), dtype)}
        return params

    @classmethod
    def randomize_ffn(cls, ffn_key, n_layer, n_embd, vocab_size, dtype):
        ratio_1_to_almost_0 = 1.0 - jnp.arange(n_layer, dtype=dtype) / (n_layer)
        params = {}
        x = jnp.arange(n_embd, dtype=dtype) / n_embd
        params['x_k'] = 1 - jnp.pow(x[None], ratio_1_to_almost_0[:, None]**4)

        hidden_sz = int(4 * n_embd)
        params['key'] = {'weight': p_orthogonal_(ffn_key, n_layer, hidden_sz, n_embd, 1.0, dtype)}
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
        n_layer = params['blocks']['att']['r_k'].shape[0]
        n_head, head_size = params['blocks']['att']['r_k'][0].shape
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
        sx = sx - x
        xk = x + sx * ffn['x_k']
        k = jnp.square(jax.nn.relu(xk @ ffn['key']['weight'].T))
        return (k @ ffn['value']['weight'].T), x[length - 1]

    @classmethod
    def inner_loop(cls, r, w, k, v, a, b, s, length, new_starts):
        w = jnp.exp(-jnp.exp(w))
        out = jnp.empty_like(r)
        out_s = s

        reset_s = jnp.zeros_like(s)
        for t in range(r.shape[0]):
            s = jax.lax.select(new_starts[t], reset_s, s)
            
            rt = jnp.expand_dims(r[t], 2)
            wt = jnp.expand_dims(w[t], 1)
            kt = jnp.expand_dims(k[t], 1)
            vt = jnp.expand_dims(v[t], 2)
            at = jnp.expand_dims(a[t], 2)
            bt = jnp.expand_dims(b[t], 1)

            sa = s@at
            s = jnp.astype(s * wt + vt @ kt + sa @ bt, s.dtype)
            out = out.at[t].set(jnp.astype((s @ rt).squeeze(2), r.dtype))
            out_s = jax.lax.select(t < length, s, out_s)
        return out_s, out

    @classmethod
    def time_mixing_seq(cls, x, state, v_first, att, length, new_starts, H, S, layer_id):
        T, C = x.shape

        sx = jnp.concatenate([state[:1], x[:-1, :]], dtype=x.dtype)
        sx = jnp.where(new_starts[:, None], jnp.zeros_like(sx), sx)
        sx = sx - x

        xr = x + sx * att['x_r']
        xw = x + sx * att['x_w']
        xk = x + sx * att['x_k']
        xv = x + sx * att['x_v']
        xa = x + sx * att['x_a']
        xg = x + sx * att['x_g']

        r = xr @ att['receptance']['weight'].T
        w = -jax.nn.softplus(-(att['w0'] + jnp.tanh(xw @ att['w1']) @ att['w2'])) - 0.5
        k = xk @ att['key']['weight'].T
        v = xv @ att['value']['weight'].T

        v_first = jnp.where(layer_id == 0, v, v_first)
        v = jnp.where(layer_id == 0, v, v + (v_first - v) * jax.nn.sigmoid(att['v0'] + (xv @ att['v1']) @ att['v2']))

        a = jax.nn.sigmoid(att['a0'] + (xa @ att['a1']) @ att['a2'])
        g = jax.nn.sigmoid(xg @ att['g1']) @ att['g2']

        kk = k * att['k_k']
        kk = kk.reshape(T, H, -1)
        kk = kk / jnp.maximum(jnp.linalg.norm(kk, axis=-1, keepdims=True), 1e-12)
        kk = kk.reshape(T, C)
        k = k * (1 + (a-1) * att['k_a'])

        state = state.at[0].set(x[length-1])
        s = jnp.reshape(state[1:, :], (H, S, S))
        # w = -jnp.exp(w)

        r, w, k, v, a_i, b_i = tuple([val.reshape(T, H, S) for val in (r, w, k, v, -kk, kk * a)])

        state_new, out = cls.inner_loop(r, w, k, v, a_i, b_i, s, length, new_starts)
        state = state.at[1:].set(state_new.reshape(S, -1))
        x = out.reshape(T, H*S)

        x = group_norm(x, num_groups=H, weight=att['ln_x']['weight'], bias=att['ln_x']['bias'], eps = 64e-5)
        x = x + (jnp.sum(r.reshape(1, T, H, -1) * k.reshape(1, T, H, -1)* att['r_k'], axis=-1, keepdims=True) * v.reshape(1, T, H, -1)).reshape(T, C)
        x = x * g
        return x @ att['output']['weight'].T, state, v_first

    @classmethod
    def forward_seq(cls, params, config, x, state, length, new_starts):
        n_layer = params['blocks']['att']['r_k'].shape[0]
        n_head, head_size = params['blocks']['att']['r_k'][0].shape
        x = layer_norm(x, params['ln0'])

        v_first = x

        @partial(jax.checkpoint,
                 policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        def block_loop(y, inputs):
            x, v_first = y
            block, state, idx = inputs
            x_new, s, v_first = cls.time_mixing_seq(layer_norm(x, block['ln1']), state[1:], v_first, block['att'], length, new_starts, n_head, head_size, idx)
            state = state.at[1:].set(s)
            x = x + x_new
            
            x_new, s = cls.channel_mixing_seq(layer_norm(x, block['ln2']), state[:1], block['ffn'], length, new_starts)
            state = state.at[0].set(s)
            x = x + x_new
            return (x, v_first), state

        (x, _), state = jax.lax.scan(block_loop, (x, v_first), (params['blocks'], state, jnp.arange(n_layer)))
        return x, state


class ScanRWKV(BaseRWKV):
    @classmethod
    def inner_loop(cls, r, w, k, v, a, b, s, length, new_starts):
        w = jnp.exp(-jnp.exp(w))
        idxes = jnp.arange(jnp.size(r, axis=0))
        reset_s = jnp.zeros_like(s)
        def scan_loop(inner_states, x):
            out_s, inner_state = inner_states
            r_t, w_t, k_t, v_t, a_t, b_t, t = x
            inner_state = jax.lax.select(new_starts[t], reset_s, inner_state)
            rt = jnp.expand_dims(r_t, 2)
            wt = jnp.expand_dims(w_t, 1)
            kt = jnp.expand_dims(k_t, 1)
            vt = jnp.expand_dims(v_t, 2)
            at = jnp.expand_dims(a_t, 2)
            bt = jnp.expand_dims(b_t, 1)

            sa = inner_state@at
            inner_state = jnp.astype(inner_state * wt + vt @ kt + sa @ bt, inner_state.dtype)
            out_t = jnp.astype((inner_state @ rt).squeeze(2), r_t.dtype)
            out_s = jax.lax.select(t < length, inner_state, out_s)
            return (out_s, inner_state), out_t
        (s, _), out = jax.lax.scan(scan_loop, (s, s), (r, w, k, v, a, b, idxes), unroll=64)
        return jnp.astype(s, r.dtype), jnp.astype(out, r.dtype)

class AssociativeScanRWKV(BaseRWKV):
    @classmethod
    def inner_loop(cls, r, w, k, v, a, b, s, length, new_starts):
        w = jnp.exp(-jnp.exp(w))

        merge_fn = jax.vmap(jax.vmap((lambda wi, ai, bi: jnp.diag(wi) + jnp.outer(ai, bi))))
        v_outer = jax.vmap(jax.vmap(jnp.outer))

        W = merge_fn(w, a, b)
        A = v_outer(v, k)
        all_A = jnp.concatenate((s[None], A), dtype=r.dtype)
        all_W = jnp.concatenate((jnp.zeros_like(W[0:1]), W))
        new_starts = jnp.concatenate((new_starts, jnp.zeros_like(new_starts[:1])))[:, None, None, None]
        all_A = jnp.where(new_starts, jnp.zeros_like(all_A), all_A)
        all_W = jnp.where(new_starts, jnp.zeros_like(all_W), all_W)
        
        def a_scan_loop(elem1, elem2):
            A1, W1 = elem1
            A2, W2 = elem2

            A12 = A2 + A1 @ W2
            W12 = W1 @ W2
            return A12, W12
        scan_s, scan_w = jax.lax.associative_scan(a_scan_loop, (all_A, all_W))

        out = (scan_s[1:] @ jnp.expand_dims(r, -1)).squeeze(-1)
        s = scan_s[length]
        return jnp.astype(s, r.dtype), jnp.astype(out, r.dtype)


import os
import ctypes
import numpy as np

UNLOADED_KERNEL = True
DO_BACKWARDS_WARNING = True

def batched_wkv7_fwd(r, w, k, v, a, b, s, length, new_starts, B):
    T = k.shape[-3]
    H = k.shape[-2]
    C = k.shape[-1]
    dtype = k.dtype
    assert all([x.dtype == dtype for x in [r, w, k, v, a, b, s]])
    out_type = jax.ShapeDtypeStruct(k.shape, k.dtype)
    s_restore_type = jax.ShapeDtypeStruct((B, H, T//16, C, C), jnp.float32)
    sa_restore_type = jax.ShapeDtypeStruct((B, T, H, C), jnp.float32)
    s_type = jax.ShapeDtypeStruct(s.shape, s.dtype)

    dtype_str = "fp32" if dtype == jnp.float32 else "bf16"

    # new_s, out = jax.ffi.ffi_call(f"wkv7-fwd-{dtype_str}", (s_type, out_type))(w, time_first, k, v, s, length, new_starts,
    #                                                B=np.uint64(B),T=np.uint64(k.shape[-3]),H=np.uint64(k.shape[-2]))

    out, s_restore, sa_restore, new_s = jax.ffi.ffi_call(f"wkv7-fwd-{dtype_str}",
                                                  (out_type, s_restore_type, sa_restore_type, s_type))(
                                                      w, r, k, v, a, b, s, length, new_starts,
                                                      B=np.uint64(B),T=np.uint64(T),H=np.uint64(H))
    return (new_s, out), (r, w, k, v, a, b, s, length, new_starts, s_restore, sa_restore)

def batched_wkv7_bwd(res, grads, B):
    r, w, k, v, a, b, s, length, new_starts, s_restore, sa_restore = res
    gs_out, gy = grads

    T = k.shape[-3]
    H = k.shape[-2]
    C = k.shape[-1]

    dtype = k.dtype

    gw_type = jax.ShapeDtypeStruct(w.shape, w.dtype)
    gr_type = jax.ShapeDtypeStruct(r.shape, r.dtype)
    gk_type = jax.ShapeDtypeStruct(k.shape, k.dtype)
    gv_type = jax.ShapeDtypeStruct(v.shape, v.dtype)
    ga_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
    gb_type = jax.ShapeDtypeStruct(b.shape, b.dtype)
    gs_type = jax.ShapeDtypeStruct(s.shape, s.dtype)


    dtype_str = "fp32" if dtype == jnp.float32 else "bf16"
    global DO_BACKWARDS_WARNING
    if DO_BACKWARDS_WARNING:
        print(f"Using {dtype_str} bwd kernel")
        print("WARNING: RWKV7 Cuda Kernels do not currently receive gradients from the output state or send gradients to the input state, making it unsuitable for state tuning. Gradients will also explode if you provided new_starts. Double check your results against AssociativeScanRWKV for correctness.")
        DO_BACKWARDS_WARNING = False

    assert T % 16 == 0, f"Must have sequence length ({T}) equal to a multiple of the chunk length (16)"

    gw, gr, gk, gv, ga, gb, gs = jax.ffi.ffi_call(f"wkv7-bwd-{dtype_str}", (gw_type, gr_type, gk_type, gv_type, ga_type, gb_type, gs_type))(
        w, r, k, v, a, b, s, length, new_starts, gy, s_restore, sa_restore, gs_out,
        B=np.uint64(B),T=np.uint64(T),H=np.uint64(H))

    # gs = jnp.zeros_like(gs)
    return gr, gw, gk, gv, ga, gb, gs, jnp.zeros_like(length), jnp.zeros_like(new_starts)


@jax.custom_batching.custom_vmap
def wkv7_fwd(r, w, k, v, a, b, s, length, new_starts):
    outs = batched_wkv7_fwd(r, w, k, v, a, b, s, jnp.uint32(length), new_starts, 1)
    return outs

@wkv7_fwd.def_vmap
def wkv7_fwd_vmap_rule(axis_size, in_batched, r, w, k, v, a, b, s, length, new_starts):

    if not in_batched[-1]:
        new_starts = jnp.repeat(new_starts[None], axis_size, axis=0)

    length = jnp.uint32(length)
    if not in_batched[-2]:
        length = jnp.repeat(length[None], axis_size, axis=0)
    
    assert all(in_batched[:-2]), f"everything must be batched {in_batched[:-2]}"
    
    return batched_wkv7_fwd(r, w, k, v, a, b, s, jnp.uint32(length), new_starts, k.shape[0]), ((True, True), tuple(in_batched) + (True, True))


@jax.custom_batching.custom_vmap
def wkv7_bwd(res, grads):
    return batched_wkv7_bwd(res, grads, 1)

@wkv7_bwd.def_vmap
def wkv7_bwd_vmap_rule(axis_size, in_batched, res, grads):
    return batched_wkv7_bwd(res, grads, axis_size), in_batched[0][:-2]

@jax.custom_vjp
def wkv7_cuda(r, w, k, v, a, b, s, length, new_starts):
    return wkv7_fwd(r, w, k, v, a, b, s, length, new_starts)[0]

wkv7_cuda.defvjp(wkv7_fwd, wkv7_bwd)



class CudaRWKV(BaseRWKV):

    @classmethod
    def get_kernels(cls):
        global UNLOADED_KERNEL
        if UNLOADED_KERNEL:
            import jaxrwkvkernel
            print("LOADING KERNELS")
            UNLOADED_KERNEL = False
            SHARED_LIBRARY = os.path.join(os.path.dirname(jaxrwkvkernel.__file__), "lib_wkv7.so")
            library = ctypes.cdll.LoadLibrary(SHARED_LIBRARY)
            jax.ffi.register_ffi_target("wkv7-fwd-fp32", jax.ffi.pycapsule(library.WKV7FwdF32), platform="CUDA")
            jax.ffi.register_ffi_target("wkv7-fwd-bf16", jax.ffi.pycapsule(library.WKV7FwdBF16), platform="CUDA")
            
            jax.ffi.register_ffi_target("wkv7-bwd-fp32", jax.ffi.pycapsule(library.WKV7BwdF32), platform="CUDA")
            jax.ffi.register_ffi_target("wkv7-bwd-bf16", jax.ffi.pycapsule(library.WKV7BwdBF16), platform="CUDA")
    
    @classmethod
    def inner_loop(cls, r, w, k, v, a, b, s, length, new_starts):
        cls.get_kernels()
        s, out = wkv7_cuda(r, w, k, v, a, b, s, length, new_starts)
        return s, out
