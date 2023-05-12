import numpy as np
import pandas as pd
import random
import os       # navigate dir. structure
from scipy.stats import truncnorm
from datetime import datetime, timedelta

def generate_gaussian_values(num_values, mean, std_dev, lower_bound, upper_bound):
    a = (lower_bound - mean) / std_dev
    b = (upper_bound - mean) / std_dev
    gaussian_values = truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=num_values)
    return gaussian_values

def generate_binary_values(num_values, percentage_zeros):
    num_zeros = int(num_values * percentage_zeros / 100)
    num_ones = num_values - num_zeros
    binary_values = np.concatenate((np.zeros(num_zeros), np.ones(num_ones)))
    np.random.shuffle(binary_values)
    return binary_values

def generate_cpu_not_idle_data(cpu_idle_data):
    # Generate corresponding values for cpu_not_idle based on cpu_idle
    return [0 if x == 1 else random.randint(0, 1) for x in cpu_idle_data]

def generate_cpu_newly_idle_data(cpu_idle_ip, cpu_not_idle_ip):
    cpu_newly_idle_data = []
    for i in range(len(cpu_idle_ip)):
        if cpu_idle_ip[i] == 1:
            cpu_newly_idle_data.append(random.randint(0, 1))
        else:
            if cpu_not_idle_ip[i] == 0:
                cpu_newly_idle_data.append(1)
            else:
                cpu_newly_idle_data.append(0)
    return cpu_newly_idle_data

def generate_load_average():
    return random.expovariate(1 / 14) * 28  # Exponential distribution with lambda=1/14, scaled by 28

def save_to_csv(data, column_name, file_name):
    df = pd.DataFrame(data, columns=[column_name])
    try:
        existing_df = pd.read_csv(file_name)
        combined_df = pd.concat([existing_df, df], axis=1)
    except FileNotFoundError:
        combined_df = df
    combined_df.to_csv(file_name, index=False) 

# file set-up
os.chdir(os.path.dirname(os.path.abspath(__file__)))
op_file_name = f'./artrawdata/op_artrawdata.csv'
if os.path.isfile(op_file_name):
    os.remove(op_file_name)

# parameters for generating artificial data
dataset_size = 86400
mean = 0
std_dev = 3
percentage_bin_dist = 50
num_cores = 14
num_threads_per_core = 2
num_numa_nodes = 2

# generate src_non_pref_nr
col_name = 'src_non_pref_nr'
binary_values = generate_binary_values(num_values = dataset_size, percentage_zeros=percentage_bin_dist)
save_to_csv(binary_values, col_name, op_file_name)
print(f"{dataset_size} elements for '{col_name}' gen")

# generate delta_hot
col_name = 'delta_hot'
binary_values = generate_binary_values(num_values = dataset_size, percentage_zeros=percentage_bin_dist)
save_to_csv(binary_values, col_name, op_file_name)
print(f"{dataset_size} elements for '{col_name}' gen")

# generate cpu_idle
col_name = 'cpu_idle'
col_cpu_idle = generate_binary_values(num_values = dataset_size, percentage_zeros=percentage_bin_dist)
save_to_csv(col_cpu_idle, col_name, op_file_name)
print(f"{dataset_size} elements for '{col_name}' gen")

# generate cpu_not_idle
col_name = 'cpu_not_idle'
col_cpu_not_idle = generate_cpu_not_idle_data(col_cpu_idle)
save_to_csv(col_cpu_not_idle, col_name, op_file_name)
print(f"{dataset_size} elements for '{col_name}' gen")

# generate cpu_newly_idle
col_name = 'cpu_newly_idle'
col_cpu_newly_idle = generate_cpu_newly_idle_data(cpu_idle_ip = col_cpu_idle, cpu_not_idle_ip = col_cpu_not_idle)
save_to_csv(col_cpu_newly_idle, col_name, op_file_name)
print(f"{dataset_size} elements for '{col_name}' gen")

# generate same_node
col_name = 'same_node'
binary_values = generate_binary_values(num_values = dataset_size, percentage_zeros=percentage_bin_dist)
save_to_csv(binary_values, col_name, op_file_name)
print(f"{dataset_size} elements for '{col_name}' gen")

# generate prefer_src
col_name = 'prefer_src'
binary_values = generate_binary_values(num_values = dataset_size, percentage_zeros=percentage_bin_dist)
save_to_csv(binary_values, col_name, op_file_name)
print(f"{dataset_size} elements for '{col_name}' gen")

# generate prefer_dst
col_name = 'prefer_dst'
binary_values = generate_binary_values(num_values = dataset_size, percentage_zeros=percentage_bin_dist)
save_to_csv(binary_values, col_name, op_file_name)
print(f"{dataset_size} elements for '{col_name}' gen")

# generate src_len
col_name = 'src_len'
lower_bound = 0
upper_bound = num_cores * num_threads_per_core * num_numa_nodes
mean = (lower_bound + upper_bound)/2
gaussian_values = generate_gaussian_values(num_values = dataset_size, mean = mean, std_dev = std_dev, lower_bound = lower_bound, upper_bound=upper_bound)
gaussian_values = np.round(gaussian_values).astype(int)
gaussian_values += 2
save_to_csv(gaussian_values, col_name, op_file_name)
print(f"{dataset_size} elements for '{col_name}' gen")

# generate src_load
col_name = 'src_load'
lambda_value = 1//num_cores
scale = num_cores ** num_threads_per_core
load_avg_values = np.random.exponential(scale=lambda_value * scale, size=dataset_size)
save_to_csv(load_avg_values, col_name, op_file_name)
print(f"{dataset_size} elements for '{col_name}' gen")

# generate dst_load
col_name = 'dst_load'
lambda_value = 1//num_cores
scale = num_cores ** num_threads_per_core
load_avg_values = np.random.exponential(scale=lambda_value * scale, size=dataset_size)
save_to_csv(load_avg_values, col_name, op_file_name)
print(f"{dataset_size} elements for '{col_name}' gen")

# generate dst_len
col_name = 'dst_len'
lower_bound = 0
upper_bound = num_cores * num_threads_per_core * num_numa_nodes
mean = (lower_bound + upper_bound)/2
gaussian_values = generate_gaussian_values(num_values = dataset_size, mean = mean, std_dev = std_dev, lower_bound = lower_bound, upper_bound=upper_bound)
gaussian_values = np.round(gaussian_values).astype(int)
save_to_csv(gaussian_values, col_name, op_file_name)
print(f"{dataset_size} elements for '{col_name}' gen")

# generate delta_faults
col_name = 'delta_faults'
lower_bound = 0
upper_bound = 10
mean = (lower_bound + upper_bound)/2
gaussian_values = generate_gaussian_values(num_values = dataset_size, mean = mean, std_dev = std_dev, lower_bound = lower_bound, upper_bound=upper_bound)
save_to_csv(gaussian_values, col_name, op_file_name)
print(f"{dataset_size} elements for '{col_name}' gen")

# generate extra_fails
col_name = 'extra_fails'
binary_values = generate_binary_values(num_values = dataset_size, percentage_zeros=percentage_bin_dist)
save_to_csv(binary_values, col_name, op_file_name)
print(f"{dataset_size} elements for '{col_name}' gen")

# generate buddy_hot
col_name = 'buddy_hot'
binary_values = generate_binary_values(num_values = dataset_size, percentage_zeros=percentage_bin_dist)
save_to_csv(binary_values, col_name, op_file_name)
print(f"{dataset_size} elements for '{col_name}' gen")