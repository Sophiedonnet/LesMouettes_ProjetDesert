import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.ndimage import convolve
import random

# Parameters definition

def get_parameters(model):
    if model.lower() == "kefi":
        return param_kefi()
    elif model.lower() == "guichard":
        return param_guichard()
    elif model.lower() == "eby":
        return param_eby()
    elif model.lower() == "forest_gap" or model.lower() == "gap" or model.lower() == "tree":
        return param_gap()
    else:
        return param_schneider()

def param_guichard():
    return {
        "δ": 0,
        "α2": 0.3,
        "α0": 0.4,
        "tau_leap": 0.5
    }

def param_gap():
    return {
        "δ": 0.3,
        "d": 0.2,
        "α": 0.2,
        "tau_leap": 0.5
    }

def param_kefi():
    return {
        "r": 0.0001,
        "d": 0.2,
        "f": 0.9,
        "m": 0.1,
        "b": 1,
        "c": 0.3,
        "delta": 0.1,
        "z": 4,
        "tau_leap": 0.5
    }

def param_schneider():
    return {
        "r": 0.0001,
        "d": 0.2,
        "f": 0.9,
        "m": 0.05,
        "b": 0.3,
        "c": 0.3,
        "delta": 0.1,
        "g0": 0.2,
        "z": 4,
        "tau_leap": 0.5
    }

def param_eby():
    return {
        "p": 0.8,
        "q": 0.8
    }

# Landscape initialization

def get_initial_lattice(model, size_landscape=100):
    if model.lower() in ["kefi", "schneider"]:
        weights = [0.1, 0.1, 0.8]
        states = [-1, 0, 1]
    elif model.lower() in ["eby", "forest_gap", "gap", "tree"]:
        weights = [0.8, 0.2]
        states = [1, 0]
    else:
        weights = [0.4, 0.4, 0.2]
        states = [0, 1, 2]
    
    ini_land = np.random.choice(states, size=(size_landscape, size_landscape), p=weights)
    return ini_land

# Model execution

def run_model(model, param, landscape, tmax=1000, keep_landscape=False, n_time_bw_snap=50, n_snapshot=25, intensity_feedback=1, burning_phase=1000):
    if model.lower() == "kefi":
        return ibm_kefi_drylands(landscape, param, tmax, keep_landscape, n_snapshot, burning_phase, n_time_bw_snap)
    elif model.lower() == "guichard":
        return ibm_guichard_mussel(landscape, param, tmax, keep_landscape, n_snapshot, burning_phase, n_time_bw_snap)
    elif model.lower() == "eby":
        return ibm_eby_drylands(landscape, param, tmax, keep_landscape, n_snapshot, burning_phase, n_time_bw_snap, intensity_feedback)
    elif model.lower() in ["forest_gap", "gap", "tree"]:
        return ibm_forest_gap(landscape, param, tmax, keep_landscape, n_snapshot, burning_phase, n_time_bw_snap)
    else:
        return ibm_schneider_drylands(landscape, param, tmax, keep_landscape, n_snapshot, burning_phase, n_time_bw_snap)

# Plotting functions

def plot_dynamics(model, d):
    if model.lower() == "guichard":
        plt.plot(d[:, 1], label="Disturbed", color="#6D6D6D", linewidth=2)
        plt.plot(d[:, 2], label="Empty", color="#88BAC1", linewidth=2)
        plt.plot(d[:, 3], label="Mussel", color="#E89090", linewidth=2)
        plt.ylim(0, 1)
    elif model.lower() in ["eby", "forest_gap", "gap", "tree"]:
        plt.plot(d[:, 0], color="lightgreen", label="vegetation")
        plt.ylim(0, 1)
    else:
        plt.plot(d[:, 0], d[:, 1], color="lightgreen", label="vegetation")
        plt.plot(d[:, 0], d[:, 2], color="orange", label="fertile")
        plt.plot(d[:, 0], d[:, 3], color="grey", label="degraded")
        plt.ylim(0, 1)
    plt.legend()
    plt.show()

def plot_landscape(model, landscape, keep_fertile=True):
    if model.lower() == "guichard":
        plt.imshow(landscape, cmap=plt.get_cmap("gray"), interpolation="nearest")
    elif model.lower() == "eby":
        plt.imshow(landscape, cmap=plt.get_cmap("gray"), interpolation="nearest")
    else:
        if keep_fertile:
            plt.imshow(landscape, cmap=plt.get_cmap("gray"), interpolation="nearest")
        else:
            landscape[landscape == 0] = -1
            plt.imshow(landscape, cmap=plt.get_cmap("gray"), interpolation="nearest")
    plt.colorbar()
    plt.show()

# Utility functions

def select_neighbor(row, col, N):
    neighbors = [
        ((row - 1) % N, col),
        ((row + 1) % N, col),
        (row, (col - 1) % N),
        (row, (col + 1) % N)
    ]
    return random.choice(neighbors)

def select_neighbor_pair(coordinate_neighbors, intensity_feedback):
    np.random.shuffle(coordinate_neighbors)
    return coordinate_neighbors[:intensity_feedback]

def get_coordinate(row, col, row_n, col_n, N):
    neighbors = [
        ((row - 1) % N, col), ((row + 1) % N, col),
        (row, (col - 1) % N), (row, (col + 1) % N),
        ((row_n - 1) % N, col_n), ((row_n + 1) % N, col_n),
        (row_n, (col_n - 1) % N), (row_n, (col_n + 1) % N)
    ]
    return np.array(neighbors)

# IBM models

def ibm_eby_drylands(landscape, param, time_t, keep_landscape=False, n_snapshot=25, burning_phase=400, n_time_bw_snap=50, intensity_feedback=1):
    p_param = param["p"]
    q_param = param["q"]
    size = landscape.shape[0]

    if keep_landscape:
        all_landscape_snap = np.zeros((size, size, n_snapshot))
        nsave = 0

    d2 = np.zeros((time_t, 2))
    d2[0, :] = [np.sum(landscape) / (size ** 2), 1 - np.sum(landscape) / (size ** 2)]

    for t in range(time_t):
        for _ in range(size * size):
            focal_i, focal_j = np.random.randint(size), np.random.randint(size)
            if landscape[focal_i, focal_j] == 1:
                neigh_i, neigh_j = select_neighbor(focal_i, focal_j, size)
                if landscape[neigh_i, neigh_j] == 0:
                    if random.random() <= p_param:
                        landscape[neigh_i, neigh_j] = 1
                    else:
                        landscape[focal_i, focal_j] = 0
                else:
                    if random.random() <= q_param:
                        neighbors = get_coordinate(focal_i, focal_j, neigh_i, neigh_j, size)
                        selected_neighbors = select_neighbor_pair(neighbors, intensity_feedback)
                        for ni, nj in selected_neighbors:
                            landscape[ni, nj] = 1
                    else:
                        landscape[focal_i, focal_j] = 0

        d2[t, 0] = np.sum(landscape) / (size ** 2)
        d2[t, 1] = 1 - d2[t, 0]

        if keep_landscape and t > burning_phase and (t - burning_phase) % n_time_bw_snap == 0:
            all_landscape_snap[:, :, nsave] = landscape
            nsave += 1

    if not keep_landscape:
        all_landscape_snap = landscape

    return d2, all_landscape_snap

# Implementing other IBM models: ibm_kefi_drylands, ibm_guichard_mussel, ibm_forest_gap, ibm_schneider_drylands
