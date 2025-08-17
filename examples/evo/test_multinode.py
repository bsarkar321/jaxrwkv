import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

from jax.experimental.multihost_utils import process_allgather, broadcast_one_to_all

# in this example, get multi-process parameters from sys.argv
import sys
proc_id = int(sys.argv[1])
num_procs = int(sys.argv[2])

# initialize the distributed system
jax.distributed.initialize('localhost:10000', num_procs, proc_id)

total_num_devices = len(jax.devices())
print("global devices", jax.devices())
print("local devices", jax.local_devices())
print("process id", jax.process_index())

mesh = jax.make_mesh((len(jax.devices()),), ('data',))
local_mesh = jax.make_mesh((len(jax.local_devices()),), ('data',))

def replicate_matrix(x):
    return jax.make_array_from_single_device_arrays(x.shape, NamedSharding(mesh, P()), [jax.device_put(x, d) for d in jax.local_devices()])

def debug_print(x):
    print("shape is", x.shape)
    for shard in x.addressable_shards:
        print(f"device {shard.device} has local data {shard.data}")


# Phases
# Step 0: Common computation across all devices: original parameters, seeds, etc

parallel_generations_per_gpu = 4
total_parallel_generations = parallel_generations_per_gpu * total_num_devices
key = replicate_matrix(jax.random.key(0))
numpy_indices = np.arange(total_parallel_generations)
global_indices = replicate_matrix(numpy_indices)
local_indices = jax.device_put(global_indices, NamedSharding(mesh, P('data')))

dataset = np.repeat(np.arange(5)[:, None], 2, axis=1)
print("true dataset", dataset)

# Step 1.0: Local computation to get tasks in numpy

def get_data_per_index(index):
    print(index, index.device)
    # print(list(index[0]))
    return jnp.array(dataset[index % dataset.shape[0]], device=index.device)

epoch = 1
unique_indices = local_indices + epoch * total_parallel_generations

current_data = jax.make_array_from_single_device_arrays((total_parallel_generations, 2), NamedSharding(mesh, P('data')), [get_data_per_index(shard.data) for shard in unique_indices.addressable_shards])

# current_data = jax.make_array_from_callback((total_parallel_generations, 2), NamedSharding(mesh, P('data')), get_data_per_index)#lambda x: get_data_per_index(epoch_num * total_parallel_generations + x))

debug_print(current_data)


# Step 1.1: Local computation to get scores

def _simple_local_computation(key, thread_idx, data, epoch):
    modified_key = jax.random.fold_in(jax.random.fold_in(key, epoch), thread_idx)
    return jax.random.normal(modified_key, data.shape) + data

simple_local_computation = jax.vmap(_simple_local_computation, in_axes=(None, 0, 0, None))

generations = simple_local_computation(key, local_indices, current_data, epoch)

print("generations")
debug_print(generations)

def get_batch_fitness(indices, full_generations):
    print("input", indices, full_generations)
    answers = dataset[indices % dataset.shape[0]]
    print(answers, full_generations)
    return jnp.array(np.mean((answers - full_generations) ** 2, axis=1), device=indices.device)

local_fitness = jax.make_array_from_single_device_arrays((total_parallel_generations,), NamedSharding(mesh, P('data')), [get_batch_fitness(shard1.data, shard2.data) for shard1, shard2 in zip(unique_indices.addressable_shards, generations.addressable_shards)])

print("local fitness")
debug_print(local_fitness)

# Step 2: Gather scores across all processes

# for i in range(num_procs):
    # print(f"from {i}:", broadcast_one_to_all(local_fitness, proc_id == i))

# global_fitness = np.array(local_fitness)
# print(global_fitness)
global_fitness = process_allgather(local_fitness)
print("global_fitness")
print(global_fitness)
# debug_print(global_fitness)

# local_cpu_fitness = jnp.array([jax.device_put(shard.data, jax.local_devices(backend='cpu')[0]) for shard in local_fitness.addressable_shards]).ravel()
# print("local cpu fitness", local_cpu_fitness)
# global_cpu_fitness = process_allgather(local_cpu_fitness)
# print("global_cpu_fitness", global_cpu_fitness)

# Step 3: (Common) computation of gradient update to params


# for shard in head_param.addressable_shards:
#     print(f"device {shard.device} has local data {shard.data}")
