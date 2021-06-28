#!/usr/bin/env python3

from os import TMP_MAX
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from  scipy.integrate import cumtrapz

# Finite size effect for the Ising Model
# João Inácio, 14th June 2021

def main():
    start = time.process_time()
    
    # Some constants and global arrays
     
    dim = "2D"
    lattice = "SS"
    NN = 4
    
    NT = 50
    Tc_Exact = 2.269
    temperatures = np.linspace(Tc_Exact-64/NT,Tc_Exact+64/NT ,NT+1)
    idx_Tc = np.where(temperatures == Tc_Exact)[0][0]
    NT = NT + 1
    
    L_vals = np.array([4, 8, 16])
        
    nu_exact = 1
    alpha_exact = 0
    beta_exact = 1/8
    gamma_exact = 7/4
    
    beta_vals = 1 / temperatures
    
    # Initialization of thermodynamic arrays
    magnetizations = list()
    M = np.zeros((len(L_vals), len(temperatures)))
    M2 = np.zeros((len(L_vals), len(temperatures)))
    M4 = np.zeros((len(L_vals), len(temperatures)))
    mod_M = np.zeros((len(L_vals), len(temperatures)))
    
    F = list()
    M_min_F = np.zeros((len(L_vals), len(temperatures)))
    min_F = np.zeros((len(L_vals), len(temperatures)))
    
    E = np.zeros((len(L_vals), len(temperatures)))
    E2 = np.zeros((len(L_vals), len(temperatures)))

    mean_C = np.zeros((len(L_vals), len(temperatures)))
    mean_S = np.zeros((len(L_vals), len(temperatures)-1))
    mean_chi = np.zeros((len(L_vals), len(temperatures))) 
    C = np.zeros((len(L_vals), len(temperatures)))
    S = np.zeros((len(L_vals), len(temperatures)))

    Tc_M_min_F = np.zeros(len(L_vals))
    Tc_mod_M = np.zeros(len(L_vals))
    idx_Tc_mod_M = np.zeros(len(L_vals))
    idx_Tc_min_F = np.zeros(len(L_vals))
    
    for idx_L in range(len(L_vals)):
        L = L_vals[idx_L]
        N_atm = L ** 2
        
        max_E = (1 / 2) * NN * N_atm
        max_M = N_atm

        NE = int(1 + (max_E / 2))
        NM = N_atm + 1
        
        energies = np.linspace(- max_E, max_E, NE)
        magnetizations.append(np.linspace(- max_M, max_M, NM))
        
        print("System -> " + dim + "_" + lattice + " | L" + str(L) + " | NT: " + str(NT))
        
        # Read all JDOS files
        file_name = "./data/JDOS_WL_Ising_" + dim + "_" + lattice + "_L" + str(L)
        JDOS = np.loadtxt(file_name + ".txt")
        log_JDOS = np.log(JDOS)
        
        # Partition function and Helmholtz free energy
        Z = np.zeros(len(temperatures))
        Z_M = np.zeros((NM, len(temperatures)))
        log_Z = np.zeros(len(temperatures))
        log_Z_M = np.zeros((NM, len(temperatures)))
        F.append(np.zeros((NM, len(temperatures))))
        
        q = 0
        hits = np.where(log_JDOS[:, q] != -np.inf)[0]
        log_Z_M[q, :] = log_JDOS[hits[0], q] - beta_vals * energies[hits[0]]
        for i in range(1, len(hits)):
            log_Z_M[q, :] += np.log(1 + np.exp(log_JDOS[hits[i], 0] - beta_vals * energies[hits[i]] - log_Z_M[q, :]))
        
        log_Z = log_Z_M[q, :]
        F[idx_L][q, :] = - temperatures * log_Z_M[q, :]
        
        for q in range(1, NM):
            hits = np.where(log_JDOS[:, q] != -np.inf)[0]

            log_Z_M[q, :] = log_JDOS[hits[0], q] - beta_vals * energies[hits[0]]
            
            for i in range(1, len(hits)):
                log_Z_M[q, :] += np.log(1 + np.exp(log_JDOS[hits[i], q] - beta_vals * energies[hits[i]] - log_Z_M[q, :]))
            
            log_Z += np.log(1 + np.exp(log_Z_M[q, :] - log_Z))
            F[idx_L][q, :] = - temperatures * log_Z_M[q, :]
        
        dif_Z  = np.zeros((NM, len(temperatures)))
        for q in range(NM):
            dif_Z[q, :] = np.exp(log_Z_M[q, :] - log_Z[:])
        dif_Z[0, :] = dif_Z[NM-1, :]
        
        # Magnetizations
        for i in range(0, len(temperatures)):
            for q in range(0, NM):
                M[idx_L][i] += magnetizations[idx_L][q] * dif_Z[q, i]
                M2[idx_L][i] += (magnetizations[idx_L][q]**2) * dif_Z[q, i]
                M4[idx_L][i] += (magnetizations[idx_L][q]**4) * dif_Z[q, i]
                mod_M[idx_L][i] += np.abs(magnetizations[idx_L][q]) * dif_Z[q, i]

        # Energies
        log_E = np.zeros(len(temperatures), dtype=np.complex64)
        log_E2 = np.zeros(len(temperatures), dtype=np.complex64)
        
        E_tmp = np.zeros(len(energies), dtype=np.complex64)
        E_tmp[:] = energies
        
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
            
            mean_C[idx_L][i] = beta_vals[i]**2 * (np.exp(log_E2[i]) - np.exp(log_E[i])**2)
            
        E[idx_L] = np.real(np.exp(log_E))
        E2[idx_L] = np.real(np.exp(log_E2))
        mean_C[idx_L] = np.real(mean_C[idx_L])
        
        # Magnetization for F minima
        for i in range(0, len(temperatures)):
            min_F[idx_L][i] = F[idx_L][0, i]
            q_min = 0
            
            for q in range(0, len(magnetizations[idx_L])):
                if F[idx_L][q, i] < min_F[idx_L][i]:
                    min_F[idx_L][i] = F[idx_L][q, i]
                    q_min = q
            
            M_min_F[idx_L][i] = np.abs(magnetizations[idx_L][q_min])
            
        # Mean magnetic susceptability, mean heat capacity and mean entropy
        for i in range(len(temperatures)):
            mean_chi[idx_L][i] = (M2[idx_L][i] - M[idx_L][i]**2) * beta_vals[i]
            
        mean_S[idx_L] = cumtrapz(mean_C[idx_L] / temperatures, x=temperatures, dx=np.abs(temperatures[0] - temperatures[1]))
        
        # Heat capacity and entropy
        F_sd = np.zeros(len(temperatures))
        F_fd = np.zeros(len(temperatures))
        h = np.abs(temperatures[1] - temperatures[2])
        for i in range(1, len(temperatures) - 1):
            F_sd[i] = (min_F[idx_L][i - 1] - 2 * min_F[idx_L][i] + min_F[idx_L][i + 1]) / h**2
            F_fd[i] = (min_F[idx_L][i + 1] - min_F[idx_L][i - 1]) / (2 * h)
            
        for i in range(len(temperatures)):
            C[idx_L][i] = - temperatures[i] * F_sd[i]
            S[idx_L][i] = - F_fd[i]
   
        # Tc apprxomation
        h = np.abs(temperatures[1] - temperatures[2])
        M_min_F_fd = np.zeros(len(temperatures))
        mod_M_fd = np.zeros(len(temperatures))
        
        for i in range(1, len(temperatures) - 1):
            M_min_F_fd[i] = (M_min_F[idx_L][i + 1] - M_min_F[idx_L][i - 1]) / (2 * h)
            mod_M_fd[i] = (mod_M[idx_L][i + 1] - mod_M[idx_L][i - 1]) / (2 * h)
        
        Tc_M_min_F[idx_L] = temperatures[np.where(M_min_F_fd == min(M_min_F_fd))[0][0]]
        idx_Tc_min_F[idx_L] = np.where(M_min_F_fd == min(M_min_F_fd))[0][0]
        Tc_mod_M[idx_L] = temperatures[np.where(mod_M_fd == min(mod_M_fd))[0][0]]
        idx_Tc_mod_M[idx_L] = np.where(mod_M_fd == min(mod_M_fd))[0][0] 
        
        # Normalize computations
        magnetizations[idx_L] /= N_atm
        mod_M[idx_L] /= N_atm
        E[idx_L] /= N_atm
        M_min_F[idx_L] /= N_atm
        F[idx_L] /= N_atm
        min_F[idx_L] /= N_atm
        C[idx_L] /= N_atm
        S[idx_L] /= N_atm
        mean_C[idx_L] /= N_atm
        mean_S[idx_L] /= N_atm
        mean_chi[idx_L] /= N_atm

    # Plots
    print()
    print(f"Tc Exact: {Tc_Exact}")
    for k in range(len(L_vals)):
        print("L{:d} -> Tc_M_min_F: {:.3f}; Tc_mod_M: {:.3f}".format(L_vals[k], Tc_M_min_F[k], Tc_mod_M[k]))

    plt.style.use('seaborn-whitegrid')
    
    fig, axs = plt.subplots(2, 2)
    for k in range(len(L_vals)):
        axs[0, 0].plot(temperatures, mod_M[k], '.-')
        axs[0, 0].set_xlabel("T")
        axs[0, 0].set_ylabel("<|M|>")
        
        axs[0, 1].plot(temperatures, E[k], '.-')
        axs[0, 1].set_xlabel("T")
        axs[0, 1].set_ylabel("<E>")
        
        axs[1, 0].plot(temperatures, M_min_F[k], '.-')
        axs[1, 0].set_xlabel("T")
        axs[1, 0].set_ylabel("M minF")
        
    for i in range(0, len(temperatures)):
        axs[1, 1].plot(magnetizations[0], F[0][:, i], '-b', lw=1)
        axs[1, 1].plot(M_min_F[0][i], min_F[0][i], '.b', ms=7.5)
    axs[1, 1].set_xlabel("M")
    axs[1, 1].set_ylabel("F")
    
    fig, axs = plt.subplots(2, 3)
    for k in range(len(L_vals)):
        axs[0, 0].plot(temperatures, C[k], '.-')
        axs[0, 0].set_xlabel("T")
        axs[0, 0].set_ylabel("C")
        
        axs[1, 0].plot(temperatures, mean_C[k], '.-')
        axs[1, 0].set_xlabel("T")
        axs[1, 0].set_ylabel("<C>")
        
        axs[0, 1].plot(temperatures, S[k], '.-')
        axs[0, 1].set_xlabel("T")
        axs[0, 1].set_ylabel("S")
        
        axs[1, 1].plot(temperatures[1:], mean_S[k], '.-')
        axs[1, 1].set_xlabel("T")
        axs[1, 1].set_ylabel("<S>")
        
        axs[0, 2].plot(temperatures, mean_chi[k], '.-')
        axs[0, 2].set_xlabel("T")
        axs[0, 2].set_ylabel("<X>")
    
    plt.figure(5)
    plt.subplot(1, 2, 1)
    
    Tc = Tc_M_min_F
    
    inv_L_vals = 1 / L_vals
    for i in range(0, len(Tc)):
        plt.plot(inv_L_vals[i], Tc[i], 'ok')
    
    plt.plot([- 0.1, np.max(inv_L_vals) + 0.1], [Tc_Exact, Tc_Exact], '-r')
    plt.xlim((- 0.1, np.max(inv_L_vals) + 0.1))
    plt.ylim((Tc_Exact - 0.5, np.max(Tc) + 0.5))
    
    a = np.polyfit(inv_L_vals, Tc, 1)
    
    x = np.linspace(0, 0.30, 100)
    y = a[0] * x + a[1]
    
    plt.plot(x, y, '-b')
    
    plt.title("Tc vs 1/L with linear regression (y={:.3f}x+{:.3f}) from Tc_M_min_F".format(a[0], a[1]))
    black_pnts = mpatches.Patch(color='black', label='Tc')
    red_line = mpatches.Patch(color='r', label='Onsager Tc')
    blue_line = mpatches.Patch(color='b', label='LinReg')
    plt.legend(handles=[red_line, black_pnts, blue_line])
    
    plt.xlabel("1/L")
    plt.ylabel("Tc")
    
    Tc_Lin_Reg = a[1]
    print("Tc by the linear regression from fitted Tc_M_min_F -> {:.3f}; error: {:.3f}".format(Tc_Lin_Reg, np.abs(Tc_Lin_Reg - Tc_Exact) / Tc_Exact))
    
    plt.subplot(1, 2, 2)
    
    Tc = Tc_mod_M
    inv_L_vals = 1 / L_vals
    for i in range(0, len(Tc)):
        plt.plot(inv_L_vals[i], Tc[i], 'ok')
        
    plt.plot([- 0.1, np.max(inv_L_vals) + 0.1], [Tc_Exact, Tc_Exact], '-r')
    plt.xlim((- 0.1, np.max(inv_L_vals) + 0.1))
    plt.ylim((Tc_Exact - 0.5, np.max(Tc) + 0.5))
    
    a = np.polyfit(inv_L_vals, Tc, 1)
    
    x = np.linspace(0, 0.30, 100)
    y = a[0] * x + a[1]
    
    plt.plot(x, y, '-b')
    
    plt.title("Tc vs 1/L with linear regression (y={:.3f}x+{:.3f}) from Tc_mod_M".format(a[0], a[1]))
    black_pnts = mpatches.Patch(color='black', label='Tc')
    red_line = mpatches.Patch(color='r', label='Onsager Tc')
    blue_line = mpatches.Patch(color='b', label='LinReg')
    plt.legend(handles=[red_line, black_pnts, blue_line])
    
    plt.xlabel("1/L")
    plt.ylabel("Tc")
    
    Tc_Lin_Reg = a[1]
    print("Tc by the linear regression from fitted Tc_mod_M -> {:.3f}; error: {:.3f}".format(Tc_Lin_Reg, np.abs(Tc_Lin_Reg - Tc_Exact) / Tc_Exact))
    
    U = np.zeros((len(L_vals), len(temperatures)))
    U = 1 - (M4 / (3 * M2**2))
    colors = ['blue', 'red', 'green', 'black']
    plt.figure(6)
    for i in range(0, len(L_vals)):
        plt.plot(temperatures, U[i, :], '.-', color=colors[i])
    
    blue_line = mpatches.Patch(color='b', label="L4")
    red_line = mpatches.Patch(color='r', label="L8")
    green_line = mpatches.Patch(color='green', label="L16")
    black_pnts = mpatches.Patch(color='black', label="L32")
    plt.legend(handles=[blue_line, red_line, green_line, black_pnts])
 
    plt.xlabel("T")
    plt.ylabel("U(T, L)")
    
    dif = U[1, :] - U[0, :]
    idx = 0
    for i in range(len(dif) - 1):
        if dif[i] > 0 and dif[i + 1] < 0:
            idx = i
            break
    new_temperatures = np.linspace(temperatures[i], temperatures[i + 1], 100)
    new_dif = np.interp(new_temperatures, temperatures, dif)
    idx = 0
    for i in range(len(new_dif) - 1):
        if new_dif[i] > 0 and new_dif[i + 1] < 0:
            idx = i
            break
    Tc_Binder = new_temperatures[idx]
    print("Tc for the Binder Cumulant -> {:.3f}; error: {:.3f}".format(Tc_Binder, np.abs(Tc_Binder - Tc_Exact) / Tc_Exact))
    
    # BETA - mean value
    
    mod_M_Tc = np.zeros(len(L_vals))
    for i in range(len(L_vals)):
        mod_M_Tc[i] = mod_M[i, idx_Tc]
    
    a = np.polyfit(np.log(L_vals), np.log(mod_M_Tc), 1) 
    beta = a[0]
    print()
    print("beta: {:.3f}; exact: {:.3f}".format(-beta, beta_exact))
    
    # ALPHA - mean value
    
    mean_C_Tc = np.zeros(len(L_vals))
    for i in range(len(L_vals)):
        mean_C_Tc[i] = mean_C[i, idx_Tc]

    a = np.polyfit(np.log(L_vals), np.log(mean_C_Tc), 1)
    alpha = a[0]
    print()
    print("alpha: {:.3f}; exact: {:.3f}".format(alpha, alpha_exact))
    
    # GAMMA - mean value    
    
    mean_chi_Tc = np.zeros(len(L_vals))
    for i in range(len(L_vals)):
        mean_chi_Tc[i] = mean_chi[i, idx_Tc]
    
    a = np.polyfit(np.log(L_vals), np.log(mean_chi_Tc), 1)
    gamma = a[0]
    print()
    print("gamma: {:.3f}; exact: {:.3f}".format(gamma, gamma_exact))
    
    print("Runtime:", time.process_time() - start)
    plt.show()
    
    
if __name__ == '__main__':
    main()
