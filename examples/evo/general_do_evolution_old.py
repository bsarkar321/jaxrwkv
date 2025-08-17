import jax
import os
from huggingface_hub.constants import HF_HOME

jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "jaxrwkvcomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp

from jaxrwkv import get_model, models
from jaxrwkv.tokenizer import LegacyWorldTokenizer

from functools import partial

import tyro
from dataclasses import dataclass

import tqdm

from typing import Optional, Literal

import time
import wandb

import numpy as np

from environments import all_tasks, validation_tasks
from rwkv7_evolution_old import RWKV7_Evolution, lora_map, FULL, LORA

import operator

from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental.multihost_utils import process_allgather


@dataclass
class Args:
    seed: int = 0
    model_choice: Literal[tuple(models.keys())] =  "7g0.1B"

    rwkv_type: str = "CudaRWKV"
    dtype: Optional[str] = None

    # parallel_generations: int = 1024
    parallel_generations_per_gpu: int = 1024
    generation_length: int = 101

    num_epochs: int = 100

    lr: float = 1e-4
    evo_sigma: float = 1e-3
    lora_dim: int = 1

    use_antithetic: bool = True

    task: Literal[tuple(all_tasks.keys())] = "fastzero"

    wandb_project: str = "evorwkv"
    wandb_name: str = "full"
    track: bool = False

    freeze_lora: bool = False
    freeze_nonlora: bool = False

    generations_per_prompt: int = 8
    group_relative_fitness: bool = False

    coord_addr: Optional[str] = None
    num_procs: Optional[int] = None
    proc_id: Optional[int] = None

args = tyro.cli(Args)

print("starting distributed init")
if args.coord_addr is not None:
    jax.distributed.initialize(args.coord_addr, args.num_procs, args.proc_id)
else:
    print("NOT DISTRIBUTED CONTEXT")
total_num_devices = len(jax.devices())
print("global devices", jax.devices())
print("local devices", jax.local_devices())
print("process id", jax.process_index())


args.proc_id = jax.process_index()
args.total_parallel_generations = total_num_devices * args.parallel_generations_per_gpu
mesh = jax.make_mesh((len(jax.devices()),), ('data',))

print()
print("per-device generations is", args.parallel_generations_per_gpu)
print("full number of generations is", args.total_parallel_generations)

# print("CURRENT MEMORY", jax.local_devices()[0].memory_stats())
RWKV, params, config, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)
legacy_tokenizer = LegacyWorldTokenizer()

if args.use_antithetic:
    assert args.parallel_generations_per_gpu % 2 == 0, "With antithetic generations, there should be even number of parallel generations"

args.prompts_per_epoch = args.total_parallel_generations // args.generations_per_prompt # TODO FIX

Task = all_tasks[args.task](tokenizer, legacy_tokenizer, args.generation_length)
validation_task = validation_tasks[args.task](tokenizer, legacy_tokenizer, args.generation_length) # TODO: validation evaluation frequency?
EvoAlgorithm = RWKV7_Evolution(args, RWKV, config)

def replicate_matrix(x):
    return jax.make_array_from_single_device_arrays(x.shape, NamedSharding(mesh, P()), [jax.device_put(x, d) for d in jax.local_devices()])

# params = jax.device_put(params, jax.local_devices()[0]) # move it to gpu (or whatever the default device is)
params = jax.tree.map(replicate_matrix, params)
# original_params = params

global_indices = replicate_matrix(np.arange(args.total_parallel_generations))
all_thread_idxes = jax.device_put(global_indices, NamedSharding(mesh, P('data')))

print("Compiling generate batch")
start_time = time.time()
# params, prompts, thread_idxes, epoch_num
# generate_batch = jax.jit(jax.vmap(EvoAlgorithm.generate_thread, in_axes=(None, 0, 0, None))).lower(params, jnp.zeros((args.parallel_generations, args.generation_length), dtype=jnp.int32), all_thread_idxes, 0).compile()
generate_batch = jax.jit(shard_map(
    jax.vmap(EvoAlgorithm.generate_thread, in_axes=(None, 0, 0, None)),
    mesh=mesh,
    in_specs=(P(), P('data'), P('data'), P()),
    out_specs=P('data')
)).lower(params, jax.ShapeDtypeStruct((args.total_parallel_generations, args.generation_length), jnp.dtype('int32')), all_thread_idxes, 0).compile()
print("Compile time", time.time() - start_time)
print("memory info")
print(generate_batch.memory_analysis())


print()
print("Compiling do update")
start_time = time.time()
# params, raw_scores, epoch_num
do_update = jax.jit(shard_map(
    EvoAlgorithm.do_update,
    mesh=mesh,
    in_specs=(P(), P(), P()),
    out_specs=P()
), donate_argnums=0).lower(params, jnp.zeros(args.total_parallel_generations), 0).compile()
print("Compile time", time.time() - start_time)
print("memory info")
print(do_update.memory_analysis())

if args.track:
    run = wandb.init(
        project=args.wandb_project,
        config=args,
        name=args.task+"_"+args.wandb_name+f"_lr={args.lr}_sigma={args.evo_sigma}_bs={args.total_parallel_generations}"
    )
else:
    print("Run name:", args.task+"_"+args.wandb_name+f"_lr={args.lr}_sigma={args.evo_sigma}_bs={args.total_parallel_generations}")

true_train_fitness_sum = 0.0


# print("CURRENT MEMORY before training", jax.local_devices()[0].memory_stats())

# for epoch in tqdm.trange(args.num_epochs):
def single_epoch(params, true_train_fitness_sum):
    
    # print("CURRENT MEMORY start of epoch", jax.local_devices()[0].memory_stats())
    start_time = time.time()
    unique_indices = jax.device_put(replicate_matrix(jnp.arange(args.prompts_per_epoch)), NamedSharding(mesh, P('data'))) + epoch * args.prompts_per_epoch
    indices = jnp.repeat(unique_indices, args.generations_per_prompt, axis=0)
    unique_prompts = jax.make_array_from_single_device_arrays((args.prompts_per_epoch, args.generation_length), NamedSharding(mesh, P('data')), [Task.get_input(shard.data) for shard in unique_indices.addressable_shards])
    # Task.get_input(unique_indices)
    batch_prompts = jnp.repeat(unique_prompts, args.generations_per_prompt, axis=0)
    prompt_processing_time = time.time() - start_time

    # print("CURRENT MEMORY start of batch", jax.local_devices()[0].memory_stats())
    start_time = time.time()
    if epoch == 0:
        print("generating batch")
    output_batch = jax.block_until_ready(generate_batch(params, batch_prompts, all_thread_idxes, epoch))
    token_generation_time = time.time() - start_time

    
    start_time = time.time()
    if epoch == 0:
        print("calculating fitness")
    # local_output_scores = jax.block_until_ready(Task.get_batch_fitness(indices, output_batch))
    _local_fitness = [jax.device_put(Task.get_batch_fitness(jax.device_put(shard1.data, jax.local_devices(backend='cpu')[0]), jax.device_put(shard2.data, jax.local_devices(backend='cpu')[0])), shard1.device) for shard1, shard2 in zip(indices.addressable_shards, output_batch.addressable_shards)]
    # for x in _local_fitness:
        # print(x.shape, x.device)
    local_fitness = jax.make_array_from_single_device_arrays((args.total_parallel_generations,), NamedSharding(mesh, P('data')), _local_fitness)

    fitness_time = time.time() - start_time

    # print("CURRENT MEMORY start of update", jax.local_devices()[0].memory_stats())
    start_time = time.time()
    if epoch == 0:
        print("gathering")
    output_scores = process_allgather(local_fitness, True)
    gather_time = time.time() - start_time


    start_time = time.time()
    if epoch == 0:
        print("updating params")
    params, parameter_differences = jax.block_until_ready(do_update(params, output_scores, epoch))
    parameter_update_time = time.time() - start_time

    # print("CURRENT MEMORY start of stats", jax.local_devices()[0].memory_stats())
    # parameter_differences = jax.tree.map(lambda x, y:jnp.mean(jnp.abs(x-y)), params, updated_params)
    lora_updates = jax.tree.reduce(operator.add, jax.tree.map(lambda x, y: x if y == LORA else 0.0, parameter_differences, lora_map)) / jax.tree.reduce(operator.add, jax.tree.map(lambda y: 1.0 if y == LORA else 0.0, lora_map))
    nonlora_updates = jax.tree.reduce(operator.add, jax.tree.map(lambda x, y: x if y == FULL else 0.0, parameter_differences, lora_map)) / jax.tree.reduce(operator.add, jax.tree.map(lambda y: 1.0 if y == FULL else 0.0, lora_map))

    # params = updated_params
    
    # parameter_differences = jax.tree.map(lambda x, y:jnp.mean(jnp.abs(x-y)), original_params, updated_params)
    # total_lora_updates = jax.tree.reduce(operator.add, jax.tree.map(lambda x, y: x if y == LORA else 0.0, parameter_differences, lora_map)) / jax.tree.reduce(operator.add, jax.tree.map(lambda y: 1.0 if y == LORA else 0.0, lora_map))
    # total_nonlora_updates = jax.tree.reduce(operator.add, jax.tree.map(lambda x, y: x if y == FULL else 0.0, parameter_differences, lora_map)) / jax.tree.reduce(operator.add, jax.tree.map(lambda y: 1.0 if y == FULL else 0.0, lora_map))
    # print(jax.tree.map(lambda x, y:jnp.mean(jnp.abs(x-y)), params, updated_params))
    # del parameter_differences

    true_train_fitness_sum += jnp.sum(output_scores).item()

    stats = {
        "avg_fitness": jnp.mean(output_scores),
        "std_fitness": jnp.std(output_scores),
        "max_fitness": jnp.max(output_scores),
        "min_fitness": jnp.min(output_scores),
        "median_fitness": jnp.median(output_scores),
        "lora_updates": lora_updates,
        "nonlora_updates": nonlora_updates,
        # "total_lora_updates": total_lora_updates,
        # "total_nonlora_updates": total_nonlora_updates,
        "prompt_preproc_time": prompt_processing_time,
        "token_gen_time": token_generation_time,
        "fitness_time": fitness_time,
        "gather_time": gather_time,
        "update_time": parameter_update_time,
        "true_train_avg_fitness": true_train_fitness_sum / ((epoch + 1) * args.total_parallel_generations)
    }
    if args.track:
        run.log(stats)
    else:
        print(f"Mean fitness: {jnp.mean(output_scores)}; std fitness: {jnp.std(output_scores)}; max fitness: {jnp.max(output_scores)}; min fitness: {jnp.min(output_scores)}; median fitness: {jnp.median(output_scores)}")
        print("mean parameter diffs")
        print("Lora modules:", lora_updates)
        print("Full modules:", nonlora_updates)
        print("Stats:")
        for k in stats:
            print(f"\t{k}: {stats[k]}")

    return params, true_train_fitness_sum

for epoch in tqdm.trange(args.num_epochs):
    params, true_train_fitness_sum = single_epoch(params, true_train_fitness_sum)

if args.track:
    run.finish()


# 3055337984
# 3055337984
# 3098466048
