#!/usr/bin/env python3

import time
import numpy as np
from matplotlib import pyplot as plt
from  scipy.integrate import cumtrapz

# Compute the therodynamics for the Ising Model
# João Inácio, 14th June 2021

def main():
    start = time.process_time()
    
    # Some variables
    kB = 1
    Tc_Onsager = 2.269
    
    dim = "2D"
    lattice = "SS"
    NN = 4
    
    L = 8
    N_atm = 1 * L ** 2
    
    max_E = (1 / 2) * NN * N_atm
    max_M = N_atm

    NE = int(1 + (max_E / 2))
    NM = N_atm + 1
    NT = 10
    
    energies = np.linspace(- max_E, max_E, NE)
    magnetizations = np.linspace(- max_M, max_M, NM)
    
    print("System -> " + dim + "_" + lattice + " | L" + str(L) + " | NT: " + str(NT))
    
    # Read all JDOS files
    
    file_name = "./data/JDOS_WL_Ising_" + dim + "_" + lattice + "_L" + str(L)
    JDOS = np.loadtxt(file_name + ".txt")
        
    # Compute thermodynamics from mean JDOS
    temperatures = np.linspace(0.5, 5, NT)
    beta_vals = 1 / (kB * temperatures)
    
    # Partition function and Helmholtz free energy
    Z = np.zeros(len(temperatures))
    Z_M = np.zeros((NM, len(temperatures)))
    F = np.zeros((NM, len(temperatures)))
    
    for q in range(0, NM):
        hits = np.where(JDOS[:, q] != 0)[0]

        for i in range(0, len(hits)):
            Z_M[q, :] += JDOS[hits[i], q] * np.exp(- beta_vals * energies[hits[i]])
        
        Z += Z_M[q, :]
        F[q, :] = - kB * temperatures * np.log(Z_M[q, :])
   
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
    
    print(E)
    print(E2)
    
    # Magnetization for F minima
    M_min_F = np.zeros(len(temperatures))
    min_F = np.zeros(len(temperatures))
    
    for i in range(0, len(temperatures)):
        min_F[i] = F[0, i]
        q_min = 0
        
        for q in range(0, len(magnetizations)):
            if F[q, i] < min_F[i]:
                min_F[i] = F[q, i]
                q_min = q
        
        M_min_F[i] = np.abs(magnetizations[q_min])
        
    # Mean magnetic susceptability, mean heat capacity and mean entropy
    mean_C = np.zeros(len(temperatures))
    mean_S = np.zeros(len(temperatures))
    
    for i in range(len(temperatures)):
        mean_C[i] = (E2[i] - E[i]**2) * beta_vals[i]
        
    mean_S = cumtrapz(mean_C / temperatures, x=temperatures, dx=np.abs(temperatures[0] - temperatures[1]))
    print(mean_C)
    # Heat capacity and entropy
    C = np.zeros(len(temperatures))
    S = np.zeros(len(temperatures))
    
    F_sd = np.zeros(len(temperatures))
    F_fd = np.zeros(len(temperatures))
    h = np.abs(temperatures[1] - temperatures[2])
    for i in range(1, len(temperatures) - 1):
        F_sd[i] = (min_F[i - 1] - 2 * min_F[i] + min_F[i + 1]) / h**2
        F_fd[i] = (min_F[i + 1] - min_F[i - 1]) / (2 * h)
        
    for i in range(len(temperatures)):
        C[i] = - temperatures[i] * F_sd[i]
        S[i] = - F_fd[i]
    
    # Tc apprxomation
    h = np.abs(temperatures[1] - temperatures[2])
    M_min_F_fd = np.zeros(len(temperatures))
    mod_M_fd = np.zeros(len(temperatures))
    
    for i in range(1, len(temperatures) - 1):
        M_min_F_fd[i] = (M_min_F[i + 1] - M_min_F[i - 1]) / (2 * h)
        mod_M_fd[i] = (mod_M[i + 1] - mod_M[i - 1]) / (2 * h)
    
    Tc_M_min_F = temperatures[np.where(M_min_F_fd == min(M_min_F_fd))[0][0]]
    Tc_mod_M = temperatures[np.where(mod_M_fd == min(mod_M_fd))[0][0]]
    Tc = Tc_M_min_F
    
    # Normalize computations
    magnetizations /= N_atm
    mod_M /= N_atm
    E /= N_atm
    M_min_F /= N_atm
    F /= N_atm
    min_F /= N_atm
    C /= N_atm
    S /= N_atm
    mean_C /= N_atm
    mean_S /= N_atm
    
    # Plots
    print("Tc_M_min_F [L{:d}]: {:.3f}".format(L, Tc_M_min_F))
    print("Tc_mod_M [L{:d}]: {:.3f}".format(L, Tc_mod_M))

    plt.style.use('seaborn-whitegrid')
    
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(temperatures, mod_M, '.-b')
    # axs[0, 0].xlabel("T")
    # axs[0, 0].ylabel("<|M|>")
    axs[0, 0].set_title("<|M|> as a function of T | L = " + str(L))
    
    axs[0, 1].plot(temperatures, E, '.-b')
    # axs[0, 1].xlabel("T")
    # axs[0, 1].ylabel("<E>")
    axs[0, 1].set_title("<E> as a function of T | L = " + str(L))
    
    axs[1, 0].plot(temperatures, M_min_F, '.-b')
    # axs[1, 0].xlabel("T/Tc")
    # axs[1, 0].ylabel("M minF")
    axs[1, 0].set_title("Magnetization for F minina as a function of T | L = " + str(L))
    
    for i in range(0, len(temperatures)):
        axs[1, 1].plot(magnetizations, F[:, i], '-b', lw=1)
        axs[1, 1].plot(M_min_F[i], min_F[i], '.b', ms=7.5)
    # axs[1, 1].xlabel("M")
    # axs[1, 1].ylabel("F")
    axs[1, 1].set_title("F as a function of M and T | L = " + str(L))
    
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(temperatures, C, '.-b')
    # axs[0, 0].xlabel("T/Tc")
    # axs[0, 0].ylabel("C")
    axs[0, 0].set_title("Heat Capacity per spin as a function of T | L = " + str(L))
    
    axs[1, 0].plot(temperatures, mean_C, '.-b')
    # axs[1, 0].xlabel("T/Tc")
    # axs[1, 0].ylabel("<C>")
    axs[1, 0].set_title("Mean Heat Capacity per spin as a function of T | L = " + str(L))
    
    axs[0, 1].plot(temperatures, S, '.-b')
    axs[0, 1].set_title("Entropy per spin as a function of T | L = " + str(L))
    
    axs[1, 1].plot(temperatures[1:], mean_S, '.-b')
    axs[1, 1].set_title("Mean Entropy per spin as a function of T | L = " + str(L))
        
    print("Script runtime: {:.4f}s".format(time.process_time() - start))
    plt.show()
    
if __name__ == '__main__':
    main()
