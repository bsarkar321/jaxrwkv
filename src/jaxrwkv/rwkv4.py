import jax
import jax.numpy as jnp

from functools import partial

from .llm import LLM

def layer_norm(x, w, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    std = jnp.sqrt(var + eps)
    return (x - mean) / std * w['weight'] + w['bias']

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
        return w

    @classmethod
    def randomize_att(cls, att_key, n_layer, n_embd, vocab_size, dtype):
        ratio_0_to_1 = jnp.arange(n_layer, dtype=dtype) / (n_layer - 1)
        ratio_1_to_almost_0 = 1.0 - jnp.arange(n_layer, dtype=dtype) / (n_layer)
        
        params = {}
        h = jnp.arange(n_embd, dtype=dtype) / (n_embd - 1)
        params['time_decay'] = jnp.astype(-5 + 8 * jnp.pow(h[None], 0.7 + 1.3 * ratio_0_to_1[:, None]), dtype)
        zigzag = ((jnp.arange(n_embd) + 1) % 3 - 1) * 0.5
        params['time_first'] = jnp.repeat(jnp.astype(jnp.log(0.3) + zigzag, dtype)[None], n_layer, axis=0)

        x = jnp.arange(n_embd, dtype=dtype) / n_embd
        params['time_mix_k'] = jnp.pow(x[None], ratio_1_to_almost_0[:, None])
        params['time_mix_v'] = jnp.pow(x[None], ratio_1_to_almost_0[:, None]) + 0.3 * ratio_0_to_1[:, None]
        params['time_mix_r'] = jnp.pow(x[None], 0.5 * ratio_1_to_almost_0[:, None])

        # key, output, receptance, value
        params['key'] = {'weight': jnp.zeros((n_layer, n_embd, n_embd), dtype=dtype)}
        params['output'] = {'weight': jnp.zeros((n_layer, n_embd, n_embd), dtype=dtype)}
        params['receptance'] = {'weight': jnp.zeros((n_layer, n_embd, n_embd), dtype=dtype)}
        
        params['value'] = {'weight': p_orthogonal_(att_key, n_layer, n_embd, n_embd, 1.0, dtype)}
        return params

    @classmethod
    def randomize_ffn(cls, ffn_key, n_layer, n_embd, vocab_size, dtype):
        ratio_1_to_almost_0 = 1.0 - jnp.arange(n_layer, dtype=dtype) / (n_layer)
        params = {}
        x = jnp.arange(n_embd, dtype=dtype) / n_embd
        params['time_mix_k'] = jnp.pow(x[None], ratio_1_to_almost_0[:, None])
        params['time_mix_r'] = jnp.pow(x[None], ratio_1_to_almost_0[:, None])

        hidden_sz = 4 * n_embd
        params['key'] = {'weight': p_orthogonal_(ffn_key, n_layer, hidden_sz, n_embd, 2.0, dtype)}
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
        state = jnp.zeros((n_layer, 5, n_embd), dtype=params['emb']['weight'].dtype)
        state = state.at[:, 4].set(-1e30)
        return state

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
        w = -jnp.exp(w)
        out = jnp.empty_like(r)
        out_s = s

        reset_s = jnp.zeros_like(s)
        reset_s = reset_s.at[2].set(-1e30)

        for t in range(r.shape[0]):
            s = jax.lax.select(new_starts[t], reset_s, s)
            aa, bb, pp = s

            ww = time_first + k[t]
            qq = jnp.maximum(pp, ww)
            e1 = jnp.exp(pp - qq)
            e2 = jnp.exp(ww - qq)
            a = e1 * aa + e2 * v[t]
            b = e1 * bb + e2
            wkv = a / b
            out = out.at[t].set(jnp.astype(r[t] * wkv, r.dtype))

            ww = pp + w
            qq = jnp.maximum(ww, k[t])
            e1 = jnp.exp(ww - qq)
            e2 = jnp.exp(k[t] - qq)

            s = jnp.astype(jnp.stack([e1 * aa + e2 * v[t], e1 * bb + e2, qq]), r.dtype)
            out_s = jax.lax.select(t < length, s, out_s)
        return out_s, out

    @classmethod
    def time_mixing_seq(cls, x, state, att, length, new_starts):
        T, C = x.shape

        sx = jnp.concatenate([state[:1], x[:-1, :]], dtype=x.dtype)
        sx = jnp.where(new_starts[:, None], jnp.zeros_like(sx), sx)
        xk = x * att['time_mix_k'] + sx * (1 - att['time_mix_k'])
        xv = x * att['time_mix_v'] + sx * (1 - att['time_mix_v'])
        xr = x * att['time_mix_r'] + sx * (1 - att['time_mix_r'])
        state = state.at[0].set(x[length-1])

        r = jax.nn.sigmoid(xr @ att['receptance']['weight'].T)
        k = xk @ att['key']['weight'].T
        v = xv @ att['value']['weight'].T

        time_decay = att['time_decay']
        
        state_new, out = cls.inner_loop(r, k, v, time_decay, att['time_first'], state[1:], length, new_starts)
        state = state.at[1:].set(state_new)
        return out @ att['output']['weight'].T, state

    @classmethod
    def forward_seq(cls, params, config, x, state, length, new_starts):
        x = layer_norm(x, params['ln0'])

        @partial(jax.checkpoint,
                 policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        def block_loop(x, inputs):
            block, state = inputs
            x_new, s = cls.time_mixing_seq(layer_norm(x, block['ln1']), state[1:], block['att'], length, new_starts)
            state = state.at[1:].set(s)
            x = x + x_new
            
            x_new, s = cls.channel_mixing_seq(layer_norm(x, block['ln2']), state[:1], block['ffn'], length, new_starts)
            state = state.at[0].set(s)
            x = x + x_new
            return x, state

        x, state = jax.lax.scan(block_loop, x, (params['blocks'], state))
        # x, state = fake_scan(block_loop, x, (params['blocks'], state))
        return x, state


class ScanRWKV(BaseRWKV):
    @classmethod
    def inner_loop(cls, r, k, v, w, time_first, s, length, new_starts):
        w = -jnp.exp(w)
        idxes = jnp.arange(jnp.size(r, axis=0))
        reset_s = jnp.zeros_like(s)
        reset_s = reset_s.at[2].set(-1e30)
        def scan_loop(inner_states, x):
            out_s, inner_state = inner_states
            r_t, k_t, v_t, t, reset = x

            inner_state = jax.lax.select(new_starts[t], reset_s, inner_state)
            aa, bb, pp = inner_state

            ww = time_first + k_t
            qq = jnp.maximum(pp, ww)
            e1 = jnp.exp(pp - qq)
            e2 = jnp.exp(ww - qq)
            a = e1 * aa + e2 * v_t
            b = e1 * bb + e2
            wkv = a / b

            out_t = jnp.astype(r_t * wkv, r.dtype)

            ww = pp + w
            qq = jnp.maximum(ww, k_t)
            e1 = jnp.exp(ww - qq)
            e2 = jnp.exp(k_t - qq)

            state = jnp.astype(jnp.stack([e1 * aa + e2 * v[t], e1 * bb + e2, qq]), r.dtype)
            out_s = jax.lax.select(t < length, state, out_s)
            return (out_s, state), out_t
        (s, _), out = jax.lax.scan(scan_loop, (s, s), (r, k, v, idxes, new_starts), unroll=32)
        return jnp.astype(s, r.dtype), jnp.astype(out, r.dtype)


class AssociativeScanRWKV(BaseRWKV):
    @classmethod
    def inner_loop(cls, r, k, v, w, time_first, s, length, new_starts):
        w = -jnp.exp(w)
        T, C = r.shape

        AA = jnp.concatenate((s[0:1], v), dtype=r.dtype)
        BB = jnp.ones_like(AA)
        BB = BB.at[0].set(s[1])
        PP = jnp.concatenate((s[2:3], k), dtype=r.dtype)
        EI = jnp.arange(T+1)[:, None]
        new_starts = jnp.concatenate((new_starts, jnp.zeros_like(new_starts[:1])))[:, None]

        AA = jnp.where(new_starts, jnp.zeros_like(AA), AA)
        BB = jnp.where(new_starts, jnp.zeros_like(BB), BB)
        PP = jnp.where(new_starts, jnp.ones_like(PP) * (-1e30), PP)

        def a_scan_loop(elem1, elem2):
            A1, B1, P1, E1, r1 = elem1
            A2, B2, P2, E2, r2 = elem2

            d = E2 - E1  # missing timesteps

            ww = P1 + d * w
            qq = jnp.maximum(ww, P2)
            AN = A1 * jnp.exp(ww - qq) + A2 * jnp.exp(P2 - qq)
            BN = B1 * jnp.exp(ww - qq) + B2 * jnp.exp(P2 - qq)
            return (
                jnp.where(r2, A2, AN),
                jnp.where(r2, B2, BN),
                jnp.where(r2, P2, qq),
                E2,
                jnp.where(r2, r2, r1)
            )

                    
        AA, BB, PP, _, _ = jax.lax.associative_scan(a_scan_loop, (AA, BB, PP, EI, new_starts))
        
        aa = AA[:T]
        bb = BB[:T]
        pp = PP[:T]

        ww = time_first[None] + k
        qq = jnp.maximum(pp, ww)
        e1 = jnp.exp(pp - qq)
        e2 = jnp.exp(ww - qq)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        out = r * (a/b)
        return jnp.stack([AA[length], BB[length], PP[length]]), out



import os
import ctypes
import numpy as np

UNLOADED_KERNEL = True
DO_BACKWARDS_WARNING = True

def batched_wkv4_fwd(k, v, w, time_first, s, length, new_starts, B):
    w = -jnp.exp(w)
    dtype = k.dtype
    assert all([x.dtype == dtype for x in [k, v, w, time_first, s]])
    out_type = jax.ShapeDtypeStruct(k.shape, k.dtype)
    s_type = jax.ShapeDtypeStruct(s.shape, s.dtype)

    dtype_str = "fp32" if dtype == jnp.float32 else "bf16"

    new_s, out = jax.ffi.ffi_call(f"wkv4-fwd-{dtype_str}", (s_type, out_type))(w, time_first, k, v, s, length, new_starts,
                                                   B=np.uint64(B),T=np.uint64(k.shape[-2]),C=np.uint64(k.shape[-1]))

    return (new_s, out), (k, v, w, time_first, s, length, new_starts)


def batched_wkv4_bwd(res, grads, B):
    k, v, w, u, s, length, new_starts = res
    assert k.shape[-1] <= 4096, "Cuda kernel compiled with length 4096 support"
    gs_new, gy = grads

    dtype = k.dtype

    gw_type = jax.ShapeDtypeStruct((B,) + w.shape, w.dtype)
    gu_type = jax.ShapeDtypeStruct((B,) + u.shape, u.dtype)
    gk_type = jax.ShapeDtypeStruct(k.shape, k.dtype)
    gv_type = jax.ShapeDtypeStruct(v.shape, v.dtype)
    gs_type = jax.ShapeDtypeStruct(s.shape, s.dtype)


    dtype_str = "fp32" if dtype == jnp.float32 else "bf16"
    global DO_BACKWARDS_WARNING
    if DO_BACKWARDS_WARNING:
        print(f"Using {dtype_str} bwd kernel")
        print("WARNING: RWKV4 Cuda Kernels do not currently receive gradients from the output state or send gradients to the input state, making it unsuitable for state tuning. Double check your results against AssociativeScanRWKV and ScanRWKV for correctness.")
        DO_BACKWARDS_WARNING = False

    gw, gu, gk, gv, gs = jax.ffi.ffi_call(f"wkv4-bwd-{dtype_str}", (gw_type, gu_type, gk_type, gv_type, gs_type))(w, u, k, v, gy,
                                                                                                                  s, length, new_starts, gs_new,
                                                   B=np.uint64(B),T=np.uint64(k.shape[-2]),C=np.uint64(k.shape[-1]))

    return gk, gv, jnp.sum(gw, axis=0), jnp.sum(gu, axis=0), gs, jnp.zeros_like(length), jnp.zeros_like(new_starts)

@jax.custom_batching.custom_vmap
def wkv4_fwd(k, v, w, time_first, s, length, new_starts):
    outs = batched_wkv4_fwd(k, v, w, time_first, s, jnp.uint32(length), new_starts, 1)
    return outs

@wkv4_fwd.def_vmap
def wkv4_fwd_vmap_rule(axis_size, in_batched, k, v, w, time_first, s, length, new_starts):
    k_batched, v_batched, w_batched, time_first_batched, s_batched, length_batched, new_starts_batched = in_batched
    should_be_batched = [k_batched, v_batched, s_batched, length_batched, new_starts_batched]
    vars_should_be_batched = [k, v, s, length, new_starts]
    k, v, s, length, new_starts = tuple([var if is_batched else jnp.repeat(var[None], axis_size, axis=0) for is_batched, var in zip(should_be_batched, vars_should_be_batched)])
    assert not w_batched, "w cannot be batched"
    assert not time_first_batched, "time_first cannot be batched"
    
    return batched_wkv4_fwd(k, v, w, time_first, s, jnp.uint32(length), new_starts, k.shape[0]), ((True, True), tuple(in_batched))


@jax.custom_batching.custom_vmap
def wkv4_bwd(res, grads):
    return batched_wkv4_bwd(res, grads, 1)

@wkv4_bwd.def_vmap
def wkv4_bwd_vmap_rule(axis_size, in_batched, res, grads):
    return batched_wkv4_bwd(res, grads, axis_size), in_batched[0]

@jax.custom_vjp
def wkv4_cuda(k, v, w, time_first, s, length, new_starts):
    return wkv4_fwd(k, v, w, time_first, s, length, new_starts)[0]

wkv4_cuda.defvjp(wkv4_fwd, wkv4_bwd)

    
class CudaRWKV(BaseRWKV):

    @classmethod
    def get_kernels(cls):
        global UNLOADED_KERNEL
        if UNLOADED_KERNEL:
            import jaxrwkvkernel
            print("LOADING KERNELS")
            UNLOADED_KERNEL = False
            SHARED_LIBRARY = os.path.join(os.path.dirname(jaxrwkvkernel.__file__), "lib_wkv4.so")
            library = ctypes.cdll.LoadLibrary(SHARED_LIBRARY)
            jax.ffi.register_ffi_target("wkv4-fwd-fp32", jax.ffi.pycapsule(library.WKV4FwdF32), platform="CUDA")
            jax.ffi.register_ffi_target("wkv4-fwd-bf16", jax.ffi.pycapsule(library.WKV4FwdBF16), platform="CUDA")
            
            jax.ffi.register_ffi_target("wkv4-bwd-fp32", jax.ffi.pycapsule(library.WKV4BwdF32), platform="CUDA")
            jax.ffi.register_ffi_target("wkv4-bwd-bf16", jax.ffi.pycapsule(library.WKV4BwdBF16), platform="CUDA")
    
    @classmethod
    def inner_loop(cls, r, k, v, w, time_first, s, length, new_starts):
        cls.get_kernels()
        s, out = wkv4_cuda(k, v, w, time_first, s, length, new_starts)
        return s, r * out
