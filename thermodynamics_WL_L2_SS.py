#!/usr/bin/env python3

import time
import numpy as np
from matplotlib import pyplot as plt
from  scipy.integrate import cumtrapz

# Compute the therodynamics for the Ising Model in a 2x2 lattice
# João Inácio, 28th June 2021

def main():
    start = time.process_time()
    
    # Some variables
    kB = 1
    Tc_Onsager = 2.269
    
    dim = "2D"
    lattice = "SS"
    NN = 4
    
    L = 2
    N_atm = L ** 2
    
    max_E = (1 / 2) * NN * N_atm
    max_M = N_atm

    NE = int(1 + (max_E / 2))
    NM = N_atm + 1
    NT = 50
    
    energies = np.linspace(- max_E, max_E, NE)
    magnetizations = np.linspace(- max_M, max_M, NM)
    
    print("System -> " + dim + "_" + lattice + " | L" + str(L) + " | NT: " + str(NT))
    
    # Read all JDOS files
    
    file_name = "./data/JDOS_WL_Ising_" + dim + "_" + lattice + "_L" + str(L)
    JDOS = np.loadtxt(file_name + ".txt")
        
    # Compute thermodynamics from mean JDOS
    temperatures = np.linspace(0.1, 10, NT)
    beta_vals = 1 / temperatures
    
    # Partition function and Helmholtz free energy
    Z = np.zeros(len(temperatures))
    F = np.zeros(len(temperatures))
    Z_M = np.zeros((NM, len(temperatures)))
    F_M = np.zeros((NM, len(temperatures)))
    
    for q in range(0, NM):
        hits = np.where(JDOS[:, q] != 0)[0]

        for i in range(0, len(hits)):
            Z_M[q, :] += JDOS[hits[i], q] * np.exp(- beta_vals * energies[hits[i]])
        
        Z += Z_M[q, :]
        F_M[q, :] = - kB * temperatures * np.log(Z_M[q, :])
    F = np.sum(F_M, 0)
    
    # Magnetizations
    M = np.zeros(len(temperatures))
    M2 = np.zeros(len(temperatures))
    M4 = np.zeros(len(temperatures))
    mod_M = np.zeros(len(temperatures))
    
    for i in range(0, len(temperatures)):        
        for q in range(0, NM):
            M[i] += magnetizations[q] * Z_M[q, i] / Z[i]
            M2[i] += (magnetizations[q]**2) * Z_M[q, i] / Z[i]
            M4[i] += (magnetizations[q]**4) * Z_M[q, i] / Z[i]
            mod_M[i] += np.abs(magnetizations[q]) * Z_M[q, i] / Z[i]
            
    # Energies
    E = np.zeros(len(temperatures))
    E2 = np.zeros(len(temperatures))
    
    for i in range(len(temperatures)):
        for j in range(len(energies)):
            E[i] += energies[j] * sum(JDOS[j, :]) * np.exp(- beta_vals[i] * energies[j]) / Z[i]
            E2[i] += (energies[j]**2) * sum(JDOS[j, :]) * np.exp(- beta_vals[i] * energies[j]) / Z[i]
   
    # Mean magnetic susceptability, mean heat capacity and mean entropy
    mean_C = np.zeros(len(temperatures))
    mean_chi = np.zeros(len(temperatures))
    
    for i in range(len(temperatures)):
        mean_C[i] = (E2[i] - E[i]**2) * (beta_vals[i]**2)
        mean_chi[i] = (M2[i] - M[i]**2) * beta_vals[i]
            
    # Tc apprxomation
    h = np.abs(temperatures[1] - temperatures[2])
    mod_M_fd = np.zeros(len(temperatures))
    
    for i in range(1, len(temperatures) - 1):
        mod_M_fd[i] = (mod_M[i + 1] - mod_M[i - 1]) / (2 * h)
    
    Tc_mod_M = temperatures[np.where(mod_M_fd == min(mod_M_fd))[0][0]]
    Tc = Tc_mod_M
    
    # Normalize computations
    mod_M /= N_atm
    E /= N_atm
    F /= N_atm
    mean_C /= N_atm
    mean_chi /= N_atm
    
    # Plots
    print("Tc_mod_M [L{:d}]: {:.3f}".format(L, Tc_mod_M))

    x_vals = np.arange(np.min(temperatures), np.max(temperatures), 0.01)

    plt.style.use('seaborn-whitegrid')
    
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].plot(temperatures, mod_M, '.-b', label="WL")
    axs[0, 0].plot(x_vals, ((4 + 2 * np.exp(8/x_vals))/(np.cosh(8/x_vals) + 3))/N_atm, '-r', label="Exato")
    axs[0, 0].set_xlabel("T")
    axs[0, 0].set_ylabel("<|M|>")
    axs[0, 0].legend()
    
    axs[0, 1].plot(temperatures, E, '.-b', label="WL")
    axs[0, 1].plot(x_vals, (-8 * np.sinh(8/x_vals) / (np.cosh(8/x_vals) + 3))/N_atm, '-r', label="Exato")
    axs[0, 1].set_xlabel("T")
    axs[0, 1].set_ylabel("<E>")
    axs[0, 1].legend()
    
    axs[1, 0].plot(temperatures, Z, '.-b', label="WL")
    axs[1, 0].plot(x_vals, (4 * np.cosh(8/x_vals) + 12), '-r', label="Exato")
    axs[1, 0].set_xlabel("T")
    axs[1, 0].set_ylabel("Z")
    axs[1, 0].legend()
    
    axs[1, 1].plot(temperatures, F, '.-b', label="WL")
    axs[1, 1].plot(x_vals, (-16 - 2*x_vals*np.log(4) - x_vals*np.log(4 + 2*np.exp(-8/x_vals)))/N_atm, '-r', label="Exato")
    axs[1, 1].set_xlabel("T")
    axs[1, 1].set_ylabel("F")
    axs[1, 1].legend()
    
    axs[0, 2].plot(temperatures, mean_C, '.-b', label="WL")
    axs[0, 2].plot(x_vals, ((64 / (x_vals**2 * (np.cosh(8/x_vals) + 3))) * (np.cosh(8/x_vals) - (np.sinh(8/x_vals)**2 / (np.cosh(8/x_vals) + 3))))/N_atm, '-r', label="Exato")
    axs[0, 2].set_xlabel("T")
    axs[0, 2].set_ylabel("<C>")
    axs[0, 2].legend() 
    
    axs[1, 2].plot(temperatures, mean_chi, '.-b', label="WL")
    axs[1, 2].plot(x_vals, ((8 * (np.exp(8/x_vals) + 1)) / (x_vals * (np.cosh(8/x_vals) + 3)))/N_atm, '-r', label="Exato")
    axs[1, 2].set_xlabel("T")
    axs[1, 2].set_ylabel("<X>")
    axs[1, 2].legend()
    
    
    print("Script runtime: {:.4f}s".format(time.process_time() - start))
    plt.show()
    
if __name__ == '__main__':
    main()


