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
    
    L = 32
    N_atm = 1 * L ** 2

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
    log_JDOS = np.zeros((NE, NM))
    log_JDOS = np.log(JDOS)
    
    # Compute thermodynamics from mean JDOS
    Tc_Exact = 2.269
    temperatures = np.linspace(Tc_Exact-64/NT,Tc_Exact+64/NT ,NT+1)
    NT = NT + 1
    beta_vals = 1 / (kB * temperatures)
    
    # Partition function and Helmholtz free energy
    Z = np.zeros(len(temperatures))
    Z_M = np.zeros((NM, len(temperatures)))
    log_Z = np.zeros(len(temperatures))
    log_Z_M = np.zeros((NM, len(temperatures)))
    F = np.zeros((NM, len(temperatures)))
    
    q = 0
    hits = np.where(log_JDOS[:, q] != -np.inf)[0]
    log_Z_M[q, :] = log_JDOS[hits[0], q] - beta_vals * energies[hits[0]]
    for i in range(1, len(hits)):
        log_Z_M[q, :] += np.log(1 + np.exp(log_JDOS[hits[i], 0] - beta_vals * energies[hits[i]] - log_Z_M[q, :]))
    
    log_Z = log_Z_M[q, :]
    F[q, :] = - temperatures * log_Z_M[q, :]
    
    for q in range(1, NM):
        hits = np.where(log_JDOS[:, q] != -np.inf)[0]

        log_Z_M[q, :] = log_JDOS[hits[0], q] - beta_vals * energies[hits[0]]
        
        for i in range(1, len(hits)):
            log_Z_M[q, :] += np.log(1 + np.exp(log_JDOS[hits[i], q] - beta_vals * energies[hits[i]] - log_Z_M[q, :]))
        
        log_Z += np.log(1 + np.exp(log_Z_M[q, :] - log_Z))
        F[q, :] = - temperatures * log_Z_M[q, :]
    
    dif_Z  = np.zeros((NM, len(temperatures)))
    for q in range(NM):
        dif_Z[q, :] = np.exp(log_Z_M[q, :] - log_Z[:])
    dif_Z[0, :] = dif_Z[NM-1, :]

    # Magnetizations
    M = np.zeros(len(temperatures))
    M2 = np.zeros(len(temperatures))
    M4 = np.zeros(len(temperatures))
    mod_M = np.zeros(len(temperatures))
    
    for i in range(0, len(temperatures)):
        for q in range(0, NM):
            M[i] += magnetizations[q] * dif_Z[q, i]
            M2[i] += (magnetizations[q]**2) * dif_Z[q, i]
            M4[i] += (magnetizations[q]**4) * dif_Z[q, i]
            mod_M[i] += np.abs(magnetizations[q]) * dif_Z[q, i]
    
    # Energies
    log_E = np.zeros(len(temperatures), dtype=np.complex64)
    log_E2 = np.zeros(len(temperatures), dtype=np.complex64)
    E = np.zeros(len(temperatures))
    E2 = np.zeros(len(temperatures))
    
    E_tmp = np.zeros(len(energies), dtype=np.complex64)
    E_tmp[:] = energies
    
    mean_C = np.zeros(len(temperatures))
    for i in range(0, len(temperatures)):
        b = 0
        k = 0
        hits = np.where(log_JDOS[b, :] != -np.inf)[0]
        
        log_E[i] = np.log(E_tmp[b]) + log_JDOS[b, hits[k]] - beta_vals[i] * E_tmp[b]
        log_E2[i] = np.log(E_tmp[b]**2) + log_JDOS[b, hits[k]] - beta_vals[i] * E_tmp[b]
        
        if len(hits) > 1:
            for k in range(1, len(hits)):
                log_E[i] += np.log(1 + np.exp(np.log(E_tmp[b]) + log_JDOS[b, hits[k]] - beta_vals[i] * E_tmp[b] - log_E[i]))
                log_E2[i] += np.log(1 + np.exp(np.log(E_tmp[b]**2) + log_JDOS[b, hits[k]] - beta_vals[i] * E_tmp[b] - log_E2[i]))

        for b in range(1, len(E_tmp)):
            hits = np.where(log_JDOS[b, :] != -np.inf)[0]

            for k in range(0, len(hits)):
                if E_tmp[b] != 0:
                    log_E[i] += np.log(1 + np.exp(np.log(E_tmp[b]) + log_JDOS[b, hits[k]] - beta_vals[i] * E_tmp[b] - log_E[i]))
                    log_E2[i] += np.log(1 + np.exp(np.log(E_tmp[b]**2) + log_JDOS[b, hits[k]] - beta_vals[i] * E_tmp[b] - log_E2[i]))
                
        log_E[i] -= log_Z[i]
        log_E2[i] -= log_Z[i]
        
        mean_C[i] = beta_vals[i]**2 * (np.exp(log_E2[i]) - np.exp(log_E[i])**2)
    
    E = np.real(np.exp(log_E))
    E2 = np.real(np.exp(log_E2))
    mean_C = np.real(mean_C)
    
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
    mean_S = np.zeros(len(temperatures))
    mean_chi = np.zeros(len(temperatures))
    
    for i in range(1, len(temperatures)):
        mean_chi[i] = (M2[i] - M[i]**2) * beta_vals[i]
    mean_S = cumtrapz(mean_C / temperatures, x=temperatures, dx=np.abs(temperatures[0] - temperatures[1]))

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
    mean_chi /= N_atm
    
    # Plots
    print("Tc_M_min_F [L{:d}]: {:.3f}".format(L, Tc_M_min_F))
    print("Tc_mod_M [L{:d}]: {:.3f}".format(L, Tc_mod_M))

    plt.style.use('seaborn-whitegrid')
    
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(temperatures, mod_M, '.-b')
    axs[0, 0].set_xlabel("T")
    axs[0, 0].set_ylabel("<|M|>")
    
    axs[0, 1].plot(temperatures, E, '.-b')
    axs[0, 1].set_xlabel("T")
    axs[0, 1].set_ylabel("<E>")
    
    axs[1, 0].plot(temperatures, M_min_F, '.-b')
    axs[1, 0].set_xlabel("T/Tc")
    axs[1, 0].set_ylabel("M minF")
    
    for i in range(0, len(temperatures)):
        axs[1, 1].plot(magnetizations, F[:, i], '-b', lw=1)
        axs[1, 1].plot(M_min_F[i], min_F[i], '.r', ms=7.5)
    axs[1, 1].set_xlabel("M")
    axs[1, 1].set_ylabel("F")
    
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(temperatures, C, '.-b')
    axs[0, 0].set_xlabel("T")
    axs[0, 0].set_ylabel("C")
    
    axs[1, 0].plot(temperatures, mean_C, '.-b')
    axs[1, 0].set_xlabel("T")
    axs[1, 0].set_ylabel("<C>")
    
    axs[0, 1].plot(temperatures, S, '.-b')
    axs[0, 1].set_xlabel("T")
    axs[0, 1].set_ylabel("S")
    
    axs[1, 1].plot(temperatures[1:], mean_S, '.-b')
    axs[1, 1].set_xlabel("T")
    axs[1, 1].set_ylabel("<S>")
        
    print("Script runtime: {:.4f}s".format(time.process_time() - start))
    plt.show()
    
if __name__ == '__main__':
    main()
