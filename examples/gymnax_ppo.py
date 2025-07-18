import jax
import os
from huggingface_hub.constants import HF_HOME

jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "jaxrwkvcomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp
import optax
from functools import partial

import numpy as np

import tyro
from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional, Tuple, Union

from jaxrwkv.auto import versions

from tqdm import trange

import time

import gymnax
from gymnax.environments import environment, spaces
import distrax

@dataclass
class Args:
    env_name: str = "CartPole-v1"
    lr: float = 2.5e-4
    num_envs: int = 4
    num_steps: int = 128
    total_timesteps: int = 500000
    update_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    debug: bool = True

    seed: int = 0
    version: Literal[tuple(versions.keys())] = "6"
    n_layer: int = 2
    n_embd: int = 128
    vocab_size: int = 2
    dtype: str = "float32"
    rwkv_type: str = "AssociativeScanRWKV"


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: jax.Array, params: Optional[environment.EnvParams] = None
    ) -> Tuple[jax.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: jax.Array,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[jax.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


class LogEnvState(NamedTuple):
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: jax.Array, params: Optional[environment.EnvParams] = None
    ) -> Tuple[jax.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: jax.Array,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[jax.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info
    

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def init_linear(key, shape, use_bias, dtype) -> any:
    key1, key2 = jax.random.split(key)
    scale = 1 / jnp.sqrt(shape[-1])
    params = {
        "weight": jax.random.uniform(key=key2, shape=shape, minval=-scale, maxval=scale, dtype=dtype)
    }
    if use_bias:
        params["bias"] = jax.random.uniform(key=key1, shape=shape[:-1], minval=-scale, maxval=scale, dtype=dtype)

    return params

def init_space_input(key, n_embd, space, dtype):
    if space['space'] == 'discrete':
        return {'discrete': jax.random.normal(key, shape=(space['n'], n_embd), dtype=dtype)}
    elif space['space'] == 'continuous':
        return {'continuous': jax.random.normal(key, shape=(np.prod(jnp.array(space['low']).shape), n_embd), dtype=dtype)}
    else:
        raise NotImplementedError('Unsupported space type')


def forward_space_input(x, params):
    if 'discrete' in params:
        return params['discrete'][x]
    elif 'continuous' in params:
        return jnp.ravel(x) @ params['continuous']
    else:
        raise NotImplementedError('Unsupported space type')


def init_space_output(key, n_embd, space, dtype):
    if space['space'] == 'discrete':
        return {'discrete': init_linear(key, (space['n'], n_embd), True, dtype)}
    elif space['space'] == 'continuous':
        return {'continuous': init_linear(key, (np.prod(jnp.array(space['low']).shape), n_embd), True, dtype)}
    elif space['space'] == 'boxdiscrete':
        high_space = jnp.ravel(jnp.array(space['high']))
        assert np.all(high_space == high_space[0])
        return {
            'boxdiscrete': init_linear(key, (np.sum(high_space + 1), n_embd), True, dtype),
            'num_values': jnp.zeros(high_space.size),
            'num_options': jnp.zeros(high_space[0] + 1)
        }
    else:
        raise NotImplementedError('Unsupported space type')

def forward_space_output(x, params):
    if 'discrete' in params:
        return x @ params['discrete']['weight'].T + params['discrete']['bias']
    elif 'continuous' in params:
        return x @ params['continuous']['weight'].T + params['continuous']['bias']
    elif 'boxdiscrete' in params:
        return x @ params['boxdiscrete']['weight'].T + params['boxdiscrete']['bias'] # B x (H0 + H1 + ...)
    else:
        raise NotImplementedError('Unsupported space type')
    

def get_space_defn(space):
    if isinstance(space, spaces.Discrete):
        return {
            'space': 'discrete',
            'start': 0,
            'n': int(space.n)
        }
    elif isinstance(space, spaces.Box) and space.dtype is jnp.float32:
        return {
            'space': 'continuous',
            'low': space.low if isinstance(space.low, jnp.ndarray) else space.low * jnp.ones(space.shape, dtype=space.dtype),
            'high': space.high if isinstance(space.high, jnp.ndarray) else space.high * jnp.ones(space.shape, dtype=space.dtype),
        }
    
    elif isinstance(space, spaces.Box):# and space.dtype.kind in ['i', 'u']:
        return {
            'space': 'boxdiscrete',
            'low': space.low if isinstance(space.low, jnp.ndarray) else space.low * jnp.ones(space.shape, dtype=space.dtype),
            'high': space.high if isinstance(space.high, jnp.ndarray) else space.high * jnp.ones(space.shape, dtype=space.dtype),
        }
    else:
        raise NotImplementedError(f'Unsupported space type {space}')

def layer_norm(x, w, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    std = jnp.sqrt(var + eps)
    return (x - mean) / std * w['weight'] + w['bias']
    
def get_actor_critic(args, rng, env, env_params):
    RWKV = getattr(versions[args.version], args.rwkv_type)
    rng, _rng = jax.random.split(rng)
    params, config = RWKV.randomize_weights(_rng, args.n_layer, args.n_embd, args.vocab_size, {}, args.dtype)
    
    # only support discrete action space
    env_act_space = get_space_defn(env.action_space(env_params))
    # only support flattened box obs space
    env_obs_space = get_space_defn(env.observation_space(env_params))

    class ActorCritic(RWKV):
        
        @classmethod
        def embed(cls, params, config, tokens):
            return jax.vmap(forward_space_input, in_axes=(0, None))(tokens, params['ac']['obs_embed'])

        @classmethod
        def outhead(cls, params, config, x):
            x = layer_norm(x, params['ln_out'])
            actor_mean = jax.nn.silu(x @ params['ac']['act_interm']['weight'].T + params['ac']['act_interm']['bias'])
            actor_mean = forward_space_output(actor_mean, params['ac']['act_head'])

            
            critic = jax.nn.silu(x @ params['ac']['value_interm']['weight'].T + params['ac']['value_interm']['bias'])
            critic = critic @ params['ac']['value_head']['weight'].T + params['ac']['value_head']['bias']
            
            return actor_mean, jnp.squeeze(critic, axis=-1)

    emb_key, act_key, value_key, act_interm_key, value_interm_key = jax.random.split(rng, 5)
    
    n_embd = params['emb']['weight'].shape[1]
    dtype = params['emb']['weight'].dtype
    params['ac'] = {
        'obs_embed': init_space_input(emb_key, n_embd, env_obs_space, dtype),
        'act_head': init_space_output(act_key, n_embd, env_act_space, dtype),
        'value_head': init_linear(value_key, (1, n_embd), True, dtype),
        'act_interm': init_linear(act_interm_key, (n_embd, n_embd), True, dtype),
        'value_interm': init_linear(value_interm_key, (n_embd, n_embd), True, dtype),
    }

    return ActorCritic, params, config


def _calculate_gae(traj_batch, last_val, last_done, args):
    def _get_advantages(carry, transition):
        gae, next_value, next_done = carry
        done, value, reward = transition.done, transition.value, transition.reward 
        delta = reward + args.gamma * next_value * (1 - next_done) - value
        gae = delta + args.gamma * args.gae_lambda * (1 - next_done) * gae
        return (gae, value, done), gae
    _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val, last_done), traj_batch, reverse=True, unroll=16)
    return advantages, advantages + traj_batch.value

def _loss_fn(params, init_hstate, traj_batch, gae, targets, forward_fn, args):
    # RERUN NETWORK
    (pi, value), _ = jax.vmap(forward_fn, in_axes=(None, 1, 0, None, 1), out_axes=1)(
        params, traj_batch.obs, init_hstate[0], 1, traj_batch.done
    )
    pi = distrax.Categorical(logits=pi)
    log_prob = pi.log_prob(traj_batch.action)
    # CALCULATE VALUE LOSS
    value_pred_clipped = traj_batch.value + (
        value - traj_batch.value
    ).clip(-args.clip_eps, args.clip_eps)
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = (
        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    )

    # CALCULATE ACTOR LOSS
    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    # jax.debug.print("ratio={ratio}", ratio=ratio.ravel())
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - args.clip_eps,
            1.0 + args.clip_eps,
        )
        * gae
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()

    total_loss = (
        loss_actor
        + args.vf_coef * value_loss
        - args.ent_coef * entropy
    )
    # jax.debug.print("losses: {loss_actor}, {value_loss}, {entropy}", loss_actor=loss_actor, value_loss=value_loss, entropy=entropy)
    return total_loss, (value_loss, loss_actor, entropy)

def _update_step(runner_state, unused, forward_fn, env, env_params, solver, args):
    # collect trajectories
    def _env_step(runner_state, unused):
        train_state, env_state, last_obs, last_done, hstate, rng = runner_state
        rng, _rng = jax.random.split(rng)

        # select action
        (pi, value), hstate = jax.vmap(forward_fn, in_axes=(None, 0, 0, None, 0))(
            train_state[0], last_obs[:, None], hstate, 1, last_done[:, None]
        )
        pi = distrax.Categorical(logits=pi)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(1),
            action.squeeze(1),
            log_prob.squeeze(1),
        )

        # step env
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, args.num_envs)
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(rng_step, env_state, action, env_params)
        transition = Transition(last_done, action, value, reward, log_prob, last_obs, info)
        runner_state = (train_state, env_state, obsv, done, hstate, rng)
        return runner_state, transition
    initial_hstate = runner_state[-2]
    runner_state, traj_batch = jax.lax.scan( # REMEMBER: time axis is 0
        _env_step, runner_state, None, args.num_steps
    )

    # CALCULATE ADVANTAGE
    train_state, env_state, last_obs, last_done, hstate, rng = runner_state
    (_, last_val), _ = jax.vmap(forward_fn, in_axes=(None, 0, 0, None, 0))(
        train_state[0], last_obs[:, None], hstate, 1, last_done[:, None]
    )
    last_val = last_val.squeeze(1)
    advantages, targets = _calculate_gae(traj_batch, last_val, last_done, args)
    
    # Update Network
    def _update_epoch(update_state, unused):
        def _update_minbatch(train_state, batch_info):
            init_hstate, traj_batch, advantages, targets = batch_info

            grad_fn = jax.value_and_grad(partial(_loss_fn, forward_fn=forward_fn, args=args), has_aux=True)
            total_loss, grads = grad_fn(
                train_state[0], init_hstate, traj_batch, advantages, targets
            )
            # train_state = train_state.apply_gradients(grads=grads)
            params, optimizer = train_state
            updates, optimizer = solver.update(grads, optimizer, params)
            params = optax.apply_updates(params, updates)
            train_state = (params, optimizer)
            return train_state, total_loss
        (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng
        ) = update_state

        rng, _rng = jax.random.split(rng)
        permutation = jax.random.permutation(_rng, args.num_envs)
        batch = (init_hstate, traj_batch, advantages, targets)

        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=1), batch
        )

        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(
                jnp.reshape(
                    x,
                    [x.shape[0], args.num_minibatches, -1] + list(x.shape[2:]),
                ), 1, 0
            ),
            shuffled_batch
        )
        # jax.debug.print("DOING MINIBATCH")
        train_state, total_loss = jax.lax.scan(
            _update_minbatch, train_state, minibatches
        )
        update_state = (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng
        )
        return update_state, total_loss
    init_hstate = initial_hstate[None]
    update_state = (
        train_state,
        init_hstate,
        traj_batch,
        advantages,
        targets,
        rng
    )
    update_state, loss_info = jax.lax.scan(
        _update_epoch, update_state, None, args.update_epochs
    )
    train_state = update_state[0]
    metric = traj_batch.info
    rng = update_state[-1]

    # Debugging mode
    if args.debug:
        def callback(info):
            return_values = info["returned_episode_returns"][info["returned_episode"]]
            timesteps = info["timestep"][info["returned_episode"]] * args.num_envs
            for t in range(len(timesteps)):
                print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
        jax.debug.callback(callback, metric)

    runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
    return runner_state, metric

    
        


def main(args):
    args.num_updates = args.total_timesteps // args.num_steps // args.num_envs
    args.minibatch_size = args.num_envs * args.num_steps // args.num_minibatches

    env, env_params = gymnax.make(args.env_name)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    
    def train(rng):
        rng, _rng = jax.random.split(rng)
        ActorCritic, network_params, config = get_actor_critic(args, _rng, env, env_params)

        # input: params, tokens, state, length (optional), new_starts (optional) -> (actor, critic), state
        forward_fn = partial(ActorCritic.forward, config=config)
        # input: params -> state
        state_fn = partial(ActorCritic.default_state, config=config)
        
        if args.anneal_lr:
            solver = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(learning_rate=lambda x: args.lr * (1.0 - (x // (args.num_minibatches * args.update_epochs)) / args.num_updates), eps=1e-5)
            )
        else:
            solver = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(args.lr, eps=1e-5)
            )

        optimizer = solver.init(network_params)
        train_state = (network_params, optimizer)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, args.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        init_hstate = jnp.repeat(state_fn(network_params)[None], args.num_envs, axis=0)
        
        # Train Loop
        
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros(args.num_envs, dtype=bool),
            init_hstate,
            _rng
        )

        runner_state, metric = jax.lax.scan(
            partial(_update_step, forward_fn=forward_fn, env=env, env_params=env_params, solver=solver, args=args), runner_state, None, args.num_updates
        )
        return {"runner_state": runner_state, "metrics": metric}
    return train


if __name__ == "__main__":
    args = tyro.cli(Args)
    train_jit = jax.jit(main(args))
    out = train_jit(jax.random.key(args.seed))
