#
# Parallel Metropolis sampling for the Ising Model
# João Inácio, June 2nd, 2021
#

import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool

def metropolis(T, N_atm, NN, NN_table, n_vals, n_equi, skip):
    """
    
    Metropolis sampling for the Ising Model
    
    Inputs: 
    T - temperature
    N_atm - Number of spins in your system
    NN - Number of nearts neightbours
    NN_table - Table of nearest neighbours
    n_vals - Number of values for averaging to get thermodynamic variables
    n_equi - Number of iterations until you get equilibrium
    skip - Get values each skip times
    
    Outputs:
    Array with 6 entries:
        1 - E_mean
        2 - E2_mean
        3 - M_mean
        4 - M2_mean
        5 - M4_mean
        6 - M_abs_mean
    
    """
    
    E_mean = 0
    E2_mean = 0
    M_mean = 0
    M2_mean = 0
    M4_mean = 0
    M_abs_mean = 0
    
    spins_vector = np.ones(N_atm)
    E = - (1.0 / 2.0) * NN * N_atm
    M = N_atm
    
    for k in range(int(n_vals * skip + n_equi)):
        for i in range(N_atm):
            flip_idx = np.random.randint(N_atm)

            delta_E = 0
            for a in range(NN):
                delta_E += - spins_vector[flip_idx] * spins_vector[int(NN_table[flip_idx, a])]
            
            E_new = E - 2 * delta_E
            M_new  = M - 2 * spins_vector[flip_idx]
            
            delta_E = - 2 * delta_E
            
            ratio = np.exp(-delta_E / T)
            
            if delta_E <= 0 or np.random.rand() < ratio:
                E = E_new
                M = M_new
                spins_vector[flip_idx] = - spins_vector[flip_idx]
        
        if k >= n_equi and np.mod(k, skip) == 0:
            E_mean += E
            E2_mean += E**2
            
            M_mean += M
            M2_mean += M**2
            M4_mean += M**4
            M_abs_mean += np.abs(M)

    return np.array([E_mean, E2_mean, M_mean, M2_mean, M4_mean, M_abs_mean]) / n_vals

def simulate(T):
    """
    
    Simulate the Ising Model with the Metropolis sampling for a given temperature T.
    
    Inputs:
    T - Temperature
    
    Outputs: 
    Array with 6 entries:
        1 - E_mean
        2 - E2_mean
        3 - M_mean
        4 - M2_mean
        5 - M4_mean
        6 - M_abs_mean 
        
    """
    
    # Define the system parameters and read files
    
    L = 4

    dim = 2
    lattice = "SS"
    N_atm = L * L
    NN = 4

    max_E = (1.0 / 2.0) * NN * N_atm
    max_M = N_atm

    NE = int(1 + (max_E / 2))
    NM = N_atm + 1

    energies_keys = np.linspace(-max_E,max_E,NE)
    magnetizations_keys = np.linspace(-max_M,max_M,NM)

    energies = dict.fromkeys(energies_keys)
    magnetizations = dict.fromkeys(magnetizations_keys)

    idx_temp=0
    for x in energies_keys:
        energies[x]=idx_temp
        idx_temp+=1

    idx_temp=0
    for x in magnetizations_keys:
        magnetizations[x]=idx_temp
        idx_temp+=1 

    NN_table_file_name = "./neighbour_tables/neighbour_table_" + str(dim) + "D_" + lattice + "_" + str(NN) + "NN_L" + str(L) + ".txt"
    NN_table = np.loadtxt(NN_table_file_name, delimiter=' ')
    
    # Computations parameters
    
    n_equi = 1E5
    n_vals = 1E3
    skip = 1
    
    # Metropolis computation
    
    met_start = time.perf_counter()
    
    ret = metropolis(T, N_atm, NN, NN_table, n_vals, n_equi, skip)
    
    wall_time = time.perf_counter() - met_start
    
    print(f"T = {T} | wall time: {wall_time}s")
    
    return ret

if __name__ == "__main__":
    
    # Defining number of processes

    n_proc = 4
    
    # Temperatures and parameters for computations
    
    NT = 16
    temperatures = np.linspace(0.1, 5, NT)
    
    E_mean = np.zeros(NT)
    E2_mean = np.zeros(NT)
    M_mean = np.zeros(NT)
    M2_mean = np.zeros(NT)
    M4_mean = np.zeros(NT)
    M_abs_mean = np.zeros(NT)
    
    # Parallel Metropolis Sampling 
    
    met_start = time.perf_counter()
    print(f'Starting computations on {n_proc} cores')

    with Pool(n_proc) as pool:
        ret = pool.map(simulate, temperatures)
    
    met_time = time.perf_counter() - met_start
    print("Total Wall Time: {:.5f}".format(met_time))
   
    for i in range(len(temperatures)):
        M_abs_mean[i] = ret[i][5]
    
    plt.figure(1)
    plt.plot(temperatures, M_abs_mean)
    
    plt.show()
    
