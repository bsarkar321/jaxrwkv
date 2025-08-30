import jax
import jax.numpy as jnp

from functools import partial

from jaxrwkv.rwkv7 import layer_norm, group_norm


UNCHANGED = 0
FULL = 1
LORA = 2

lora_map = {'blocks': {
    'att': {'a0': FULL, 'a1': LORA, 'a2': LORA, 'g1': LORA, 'g2': LORA, 'k_a': FULL, 'k_k': FULL, 'key': {'weight': LORA},
            'ln_x': {'bias': FULL, 'weight': FULL}, 'output': {'weight': LORA},
            'r_k': FULL, # LORA EXCEPTION
            'receptance': {'weight': LORA},
            'v0': FULL, 'v1': LORA, 'v2': LORA,
            'value': {'weight': LORA},
            'w0': FULL, 'w1': LORA, 'w2': LORA, 'x_a': FULL, 'x_g': FULL, 'x_k': FULL, 'x_r': FULL, 'x_v': FULL, 'x_w': FULL},
    'ffn': {'key': {'weight': LORA}, 'value': {'weight': LORA}, 'x_k': FULL},
    'ln1': {'bias': FULL, 'weight': FULL}, 'ln2': {'bias': FULL, 'weight': FULL}},
    'emb': {'weight': UNCHANGED},
    'head': {'weight': UNCHANGED},
    'ln0': {'bias': FULL, 'weight': FULL},
    'ln_out': {'bias': FULL, 'weight': FULL}
}

def fold_in_helper(key, epoch, true_thread_idx):
    return jax.random.fold_in(jax.random.fold_in(key, epoch), true_thread_idx)

def do_normalize(x):
    return x / (jnp.linalg.norm(x, axis=0) + 1e-5)

def get_lora_update_params(iterinfo, param, key, lora_dim, lora_config):
    epoch, sigma, true_thread_idx = iterinfo

    outer_epoch = 0 if len(lora_config) <= 2 or lora_config[2] == '0' else (epoch // int(lora_config[2:]))
    
    a, b = param.shape
    lora_params = jax.random.normal(fold_in_helper(key, epoch, true_thread_idx), (a+b, lora_dim), dtype=param.dtype)
    if lora_config[0] == 'S':
        lora_const_params = jax.random.normal(fold_in_helper(key, outer_epoch, true_thread_idx), (a+b, lora_dim), dtype=param.dtype)
    else:
        lora_const_params = jax.random.normal(jax.random.fold_in(key, outer_epoch), (a+b, lora_dim), dtype=param.dtype)

    nonzero_A = (lora_config[0] == 'A') or (lora_config[0] in ['C', 'S'] and outer_epoch % 2 == 0)
    
    B = lora_params[:b]
    A = lora_params[b:]

    Bc = lora_const_params[:b]
    Ac = lora_const_params[b:]
    if lora_config[1] == '1':
        Bc = jnp.ones_like(Bc)
        Ac = jnp.ones_like(Ac)

    A = jax.lax.select(nonzero_A, jnp.zeros_like(A), A)
    B = jax.lax.select(nonzero_A, B, jnp.zeros_like(B))

    Ac = jax.lax.select(nonzero_A, do_normalize(Ac) * jnp.sqrt(a), jnp.zeros_like(Ac))
    Bc = jax.lax.select(nonzero_A, jnp.zeros_like(Bc), do_normalize(Bc) * jnp.sqrt(b))
    
    return A, B, Ac, Bc

def get_lora_params(iterinfo, param, key, lora_dim, lora_config):
    epoch, sigma, true_thread_idx = iterinfo
    A, B, Ac, Bc = get_lora_update_params(iterinfo, param, key, lora_dim, lora_config)
    return (A * sigma + Ac).T, B * sigma + Bc

def get_evo_class(args, RWKV):
    def evo_lora2(iterinfo, M, param, key):
        if args.freeze_lora:
            return M@param

        A, B = get_lora_params(iterinfo, param, key, args.lora_dim, args.lora_config)
        return M @ param + (M @ A.T) @ B.T

    def evo_lora(iterinfo, M, param, key):
        if args.freeze_lora:
            return M@param.T
        
        A, B = get_lora_params(iterinfo, param, key, args.lora_dim, args.lora_config)
        return M @ param.T + (M @ B) @ A

    def evo(iterinfo, param, key):
        if args.freeze_nonlora:
            return param

        epoch, sigma, true_thread_idx = iterinfo
        return param + jax.random.normal(fold_in_helper(key, epoch, true_thread_idx), param.shape, dtype=param.dtype) * sigma # r_k is exception

    
    class EvoRWKV(RWKV):

        @classmethod
        def evo_channel_mixing_seq(cls, x, state, ffn, key_ffn, length, new_starts, iterinfo):
            sx = jnp.concatenate([state, x[:-1, :]], dtype=x.dtype)
            sx = jnp.where(new_starts[:, None], jnp.zeros_like(sx), sx)
            sx = sx - x
            xk = x + sx * evo(iterinfo, ffn['x_k'], key_ffn['x_k'])
            k = jnp.square(jax.nn.relu(evo_lora(iterinfo, xk, ffn['key']['weight'], key_ffn['key']['weight']))) # LORA
            return evo_lora(iterinfo, k, ffn['value']['weight'], key_ffn['value']['weight']), x[length - 1] # LORA

        @classmethod
        def evo_time_mixing_seq(cls, x, state, v_first, att, key_att, length, new_starts, H, S, layer_id, iterinfo):
            T, C = x.shape

            sx = jnp.concatenate([state[:1], x[:-1, :]], dtype=x.dtype)
            sx = jnp.where(new_starts[:, None], jnp.zeros_like(sx), sx)
            sx = sx - x

            xr = x + sx * evo(iterinfo, att['x_r'], key_att['x_r'])
            xw = x + sx * evo(iterinfo, att['x_w'], key_att['x_w'])
            xk = x + sx * evo(iterinfo, att['x_k'], key_att['x_k'])
            xv = x + sx * evo(iterinfo, att['x_v'], key_att['x_v'])
            xa = x + sx * evo(iterinfo, att['x_a'], key_att['x_a'])
            xg = x + sx * evo(iterinfo, att['x_g'], key_att['x_g'])

            r = evo_lora(iterinfo, xr, att['receptance']['weight'], key_att['receptance']['weight']) # LORA
            w = -jax.nn.softplus(-(evo(iterinfo, att['w0'], key_att['w0']) + evo_lora2(iterinfo, jnp.tanh(evo_lora2(iterinfo, xw, att['w1'], key_att['w1'])), att['w2'], key_att['w2']))) - 0.5 # LORA2, LORA2
            k = evo_lora(iterinfo, xk, att['key']['weight'], key_att['key']['weight']) # LORA
            v = evo_lora(iterinfo, xv, att['value']['weight'], key_att['value']['weight']) # LORA

            v_first = jnp.where(layer_id == 0, v, v_first)
            v = jnp.where(layer_id == 0, v, v + (v_first - v) * jax.nn.sigmoid(
                evo(iterinfo, att['v0'], key_att['v0']) + evo_lora2(iterinfo, (evo_lora2(iterinfo, xv, att['v1'], key_att['v1'])), att['v2'], key_att['v2'])
            ))

            a = jax.nn.sigmoid(evo(iterinfo, att['a0'], key_att['a0']) + evo_lora2(iterinfo, (evo_lora2(iterinfo, xa, att['a1'], key_att['a1'])), att['a2'], key_att['a2']))
            g = evo_lora2(iterinfo, jax.nn.sigmoid(evo_lora2(iterinfo, xg, att['g1'], key_att['g1'])), att['g2'], key_att['g2'])

            kk = k * evo(iterinfo, att['k_k'], key_att['k_k'])
            kk = kk.reshape(T, H, -1)
            kk = kk / jnp.maximum(jnp.linalg.norm(kk, axis=-1, keepdims=True), 1e-12)
            kk = kk.reshape(T, C)
            k = k * (1 + (a-1) * evo(iterinfo, att['k_a'], key_att['k_a']))

            state = state.at[0].set(x[length-1])
            s = jnp.reshape(state[1:, :], (H, S, S))

            r, w, k, v, a_i, b_i = tuple([val.reshape(T, H, S) for val in (r, w, k, v, -kk, kk * a)])

            state_new, out = cls.inner_loop(r, w, k, v, a_i, b_i, s, length, new_starts)
            state = state.at[1:].set(state_new.reshape(S, -1))
            x = out.reshape(T, H*S)

            x = group_norm(x, num_groups=H, weight=evo(iterinfo, att['ln_x']['weight'], key_att['ln_x']['weight']), bias=evo(iterinfo, att['ln_x']['bias'], key_att['ln_x']['bias']), eps = 64e-5)
            x = x + (jnp.sum(r.reshape(1, T, H, -1) * k.reshape(1, T, H, -1) * evo(iterinfo, att['r_k'], key_att['r_k']), axis=-1, keepdims=True) * v.reshape(1, T, H, -1)).reshape(T, C)
            x = x * g
            return evo_lora(iterinfo, x, att['output']['weight'], key_att['output']['weight']), state, v_first # LORA

        @classmethod
        def evo_forward_seq(cls, params, key_params, config, x, state, length, new_starts, iterinfo):
            n_layer = params['blocks']['att']['r_k'].shape[0]
            n_head, head_size = params['blocks']['att']['r_k'][0].shape
            x = layer_norm(x, jax.tree.map(partial(evo, iterinfo), params['ln0'], key_params['ln0']))

            v_first = x

            @partial(jax.checkpoint,
                     policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
            def block_loop(y, inputs):
                x, v_first = y
                block, key_block, state, idx = inputs
                x_new, s, v_first = cls.evo_time_mixing_seq(layer_norm(x, jax.tree.map(partial(evo, iterinfo), block['ln1'], key_block['ln1'])), state[1:], v_first, block['att'], key_block['att'], length, new_starts, n_head, head_size, idx, iterinfo)
                state = state.at[1:].set(s)
                x = x + x_new

                x_new, s = cls.evo_channel_mixing_seq(layer_norm(x, jax.tree.map(partial(evo, iterinfo), block['ln2'], key_block['ln2'])), state[:1], block['ffn'], key_block['ffn'], length, new_starts, iterinfo)
                state = state.at[0].set(s)
                x = x + x_new
                return (x, v_first), state

            (x, _), state = jax.lax.scan(block_loop, (x, v_first), (params['blocks'], key_params['blocks'], state, jnp.arange(n_layer)))
            return x, state

        @classmethod
        def evo_forward(cls, params, key_params, tokens, state, iterinfo, length=None, new_starts=None, config=None):
            """
            Forward pass on a single stream of tokens
            """
            tokens = jnp.array(tokens)
            x = cls.embed(params, config, tokens) # doesn't include key_params
            T, D = x.shape
            if length is None:
                length = T
            if new_starts is None:
                new_starts = jnp.zeros((T,), dtype=jnp.bool)
            x, state = cls.evo_forward_seq(params, key_params, config, x, state, length, new_starts, iterinfo)
            x = cls.outhead(params, config, x) # doesn't include key_params
            return x, state

    return EvoRWKV

def get_model_tree_keys(params, base_model_key):
    vals, treedef = jax.tree.flatten(params)
    all_keys = jax.random.split(base_model_key, len(vals))
    partial_key_tree = jax.tree.unflatten(treedef, all_keys)
    n_layer = params['blocks']['att']['r_k'].shape[0]
    partial_key_tree['blocks'] = jax.tree.map(lambda x: jax.random.split(x, n_layer), partial_key_tree['blocks'])
    return partial_key_tree


def RWKV7_Evolution(args, RWKV, config):
    args = args
    RWKV = RWKV
    config = config
    EvoRWKV = get_evo_class(args, RWKV)

    def _single_get_iterinfo(thread_idx, epoch):
        sigma_antithetic = jnp.where(thread_idx % 2 == 0, args.evo_sigma, -args.evo_sigma) if args.use_antithetic else args.evo_sigma
        true_thread_idx = thread_idx // 2 if args.use_antithetic else thread_idx
        return epoch, sigma_antithetic, true_thread_idx

    def forward_and_sample(model, model_keys, input_token, input_state, generation_key, iterinfo):
        print("compiling forward and sample")
        gen_key, _gen_key = jax.random.split(generation_key)
        generated_outs, generated_state = EvoRWKV.evo_forward(model, model_keys, input_token, input_state, iterinfo)
        sampled_tok = jax.random.categorical(_gen_key, generated_outs[-1])
        return sampled_tok, generated_state, gen_key

    def generate_thread(params, prompt, thread_idx, epoch_num):
        print("compiling generate_batch")
        
        key = jax.random.key(args.seed)
        base_gen_key, base_model_key = jax.random.split(key)
        # start_gen_key = jax.random.fold_in(jax.random.fold_in(base_gen_key, epoch_num), thread_idx)
        start_gen_key = fold_in_helper(base_gen_key, epoch_num, thread_idx)
        model_tree_key = get_model_tree_keys(params, base_model_key)
        
        iterinfo = _single_get_iterinfo(thread_idx, epoch_num)
        def inner_scan(carry, input_token):
            tok, state, gen_key = carry
            true_input = jnp.where(input_token == 0, tok, input_token)
            tok, state, gen_key = forward_and_sample(params, model_tree_key, true_input, state, gen_key, iterinfo)
            return (tok, state, gen_key), true_input

        init_state = RWKV.default_state(params, config=config)

        _, out_tokens = jax.lax.scan(inner_scan, (0, init_state, start_gen_key), prompt)
        return out_tokens

    def _simple_full_update(param, key, scores, iterinfo):
        if args.freeze_nonlora:
            return jnp.zeros_like(param)

        epoch, sigma, true_thread_idx = iterinfo
        noises = jax.vmap(partial(jax.random.normal, shape=param.shape, dtype=param.dtype))(jax.vmap(fold_in_helper, in_axes=(None, 0, 0))(key, epoch, true_thread_idx))
        broadcasted_scores = jnp.reshape(scores, scores.shape + (1,) * len(param.shape))
        broadcasted_sigma = jnp.reshape(sigma, sigma.shape + (1,) * len(param.shape))
        # return jnp.astype(param + lr * jnp.mean(broadcasted_scores * noises / broadcasted_sigma, axis=0), param.dtype)
        return jnp.astype(jnp.mean(broadcasted_scores * noises / broadcasted_sigma, axis=0), param.dtype)

    def _simple_lora_update(param, key, scores, iterinfo):
        if args.freeze_lora:
            return jnp.zeros_like(param)
        # a, b = param.shape
        # epoch, sigma, true_thread_idx = iterinfo
        # true_key = jax.vmap(fold_in_helper, in_axes=(None, 0, 0))(key, epoch, true_thread_idx)
        # noises = jax.vmap(partial(jax.random.normal, shape=(a+b, args.lora_dim), dtype=param.dtype))(true_key)
        # Bs = noises[:, :b]
        # As = jnp.ones_like(noises[:, b:])
        epoch, sigma, true_thread_idx = iterinfo
        At, Bt, Ac, Bc = jax.vmap(partial(get_lora_update_params, lora_dim=args.lora_dim, lora_config=args.lora_config), in_axes=(0, None, None))(iterinfo, param, key)
        broadcasted_scores = jnp.reshape(scores, scores.shape + (1,) * len(param.shape))
        broadcasted_sigma = jnp.reshape(sigma, sigma.shape + (1,) * len(param.shape))# / lr
        preB = broadcasted_scores / broadcasted_sigma * Bt
        preA = broadcasted_scores / broadcasted_sigma * At
        # preA = As # not adapting

        if args.lora_config[0] == 'S':
            A = Ac + preA
            B = Bc + preB
            print("A shape", A.shape, "B shape", B.shape)
            actual_grad = jnp.mean(A @ B.mT, axis=0) # B x N x 1 @ B x 1 x N
        else:
            B = jnp.mean(Bc + preB, axis=0)
            A = jnp.mean(Ac + preA, axis=0) # not adapting

            if args.muon:
                B = jax.nn.standardize(B, axis=0)
                A = jax.nn.standardize(A, axis=0)

            actual_grad = A @ B.mT
        return jnp.astype(actual_grad, param.dtype)
    

    def get_gradient(params, raw_scores, epoch_num):
        print("compiling do_update")

        print(raw_scores.shape)

        if args.group_relative_fitness:
            group_scores = raw_scores.reshape((-1, args.generations_per_prompt))
            true_scores = (group_scores - jnp.mean(group_scores, axis=-1, keepdims=True)).ravel()
        else:
            true_scores = (raw_scores - jnp.mean(raw_scores, keepdims=True)) / jnp.sqrt(jnp.var(raw_scores, keepdims=True) + 1e-5)

        key = jax.random.key(args.seed)
        base_gen_key, base_model_key = jax.random.split(key)
        # start_gen_key = fold_in_helper(base_gen_key, epoch, thread_idx)
        model_tree_key = get_model_tree_keys(params, base_model_key)

        batch_iterinfo = jax.vmap(_single_get_iterinfo, in_axes=(0, None))(jnp.arange(raw_scores.size), epoch_num)
        
        def inner_update(param, model_key, lora_map_ans):
            if lora_map_ans == UNCHANGED:
                return jnp.zeros_like(param)#, 0.0

            update_fn = [_simple_full_update, _simple_lora_update][lora_map_ans - 1]

            if len(model_key.shape) == 0:
                new_grad = update_fn(param, model_key, true_scores, batch_iterinfo)
            else:
                new_grad = jax.lax.scan(lambda _, x: (0, update_fn(x[0], x[1], true_scores, batch_iterinfo)), 0, xs=(param, model_key))[1]
            # gradients are negative for optax
            return -new_grad#, jnp.mean(jnp.abs(param-new_param))

        # merged_params_l1 = jax.tree.map(inner_update, params, model_tree_key, lora_map)
        # return jax.tree.map(lambda x: x[0], merged_param_l1), jax.tree.map(lambda x: x[1].astype(jnp.float32), merged_param_l1)
        # return jax.tree.transpose(jax.tree.structure(params), None, merged_params_l1)
        return jax.tree.map(inner_update, params, model_tree_key, lora_map)

    return generate_thread, get_gradient
