import numpy as np

EPSILON = 1e-10

def compute_stats(u_data, c_data, t_values, metadata, max_time_diff = 14, sample_rate = 0.1, use_metadata_stats = False, use_time_norm = False):
    """
    Compute statistics for normalization

    Parameters:
            u_data (numpy array): The input data, shape: [num_samples, num_timesteps, num_nodes, num_active_vars]
            c_data (numpy array): The metadata, shape: [num_samples, num_timesteps, num_nodes, num_metadata_vars]
            t_values (numpy array): The time values, shape: [num_samples, num_timesteps]
            max_time_diff (int): The maximum time difference
            use_metadata_stats (bool): Whether to use metadata statistics
            sample_rate (float): The sample rate
    """
    stats = {}
    stats["u"] = {}
    stats["res"] = {}
    stats["der"] = {}
    stats["start_time"] = {}
    stats["time_diffs"] = {}

    num_samples, num_timesteps, num_nodes, num_vars = u_data.shape

    num_timesteps = max_time_diff
    t_values = t_values[:num_timesteps + 1] # Shape: [num_timesteps + 1]

    t_in_indices = []
    t_out_indices = []
    for lag in range(2, num_timesteps + 1, 2):  # Even lags from 2 to 14
        num_pairs = (num_timesteps - lag) // 2 + 1
        for i in range(0, num_timesteps - lag + 1, 2):
            t_in_idx = i
            t_out_idx = i + lag
            t_in_indices.append(t_in_idx)
            t_out_indices.append(t_out_idx)
    
    t_in_indices = np.array(t_in_indices)
    t_out_indices = np.array(t_out_indices)
    num_time_pairs = len(t_in_indices)
    total_pairs = sample_rate * num_samples * num_time_pairs # The total pair for calculating statistics

    time_diffs = t_values[t_out_indices] - t_values[t_in_indices] # Shape: [num_time_pairs]
    start_times = t_values[t_in_indices] # Shape: [num_time_pairs]

    stats["u"]["mean"] = np.mean(u_data, axis=(0,1,2))  # Shape: [num_active_vars]
    stats["u"]["std"] = np.std(u_data, axis=(0,1,2)) + EPSILON
    if use_metadata_stats:
        stats["u"]["mean"] = np.array(metadata.global_mean)
        stats["u"]["std"] = np.array(metadata.global_std)
    if c_data is not None:
        stats["c"] = {}
        stats["c"]["mean"] = np.mean(c_data, axis=(0,1,2))
        stats["c"]["std"] = np.std(c_data, axis=(0,1,2)) + EPSILON
    
    stats["start_time"]["mean"] = np.mean(start_times)
    stats["start_time"]["std"] = np.std(start_times) + EPSILON
    stats["time_diffs"]["mean"] = np.mean(time_diffs)
    stats["time_diffs"]["std"] = np.std(time_diffs) + EPSILON
    if not use_time_norm:
        stats["start_time"]["mean"] = 0.0
        stats["start_time"]["std"] = 1.0
        stats["time_diffs"]["mean"] = 0.0
        stats["time_diffs"]["std"] = 1.0
    
    u_data_sample = u_data[:int(sample_rate*num_samples)] # Shape: [num_samples, num_timesteps, num_nodes, num_active_vars]

    res = np.ascontiguousarray(u_data_sample[:,t_in_indices] - u_data_sample[:,t_out_indices]) # Shape: [num_samples, num_time_pairs, num_nodes, num_active_vars]
    stats["res"]["mean"] = np.mean(res, axis=(0,1,2))  # Shape: [num_active_vars]
    stats["res"]["std"] = np.std(res, axis=(0,1,2)) + EPSILON

    der = np.ascontiguousarray((u_data_sample[:,t_out_indices] - u_data_sample[:,t_in_indices]) / time_diffs[None, :, None, None]) # Shape: [num_samples, num_time_pairs, num_nodes, num_active_vars]
    stats["der"]["mean"] = np.mean(der, axis=(0,1,2))  # Shape: [num_active_vars]
    stats["der"]["std"] = np.std(der, axis=(0,1,2)) + EPSILON

    return stats

    
    

