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
    kB = 1
    
    dim = "3D"

    if dim == "2D":
        lattice = "SS"
        NN = 4
        NT = 100
        temperatures = np.linspace(1, 5, NT, dtype=np.longdouble)
        L_vals = np.array([4, 8, 16])
        Tc_Exact = 2.269
        
        nu_exact = 1
        alpha_exact = 0
        beta_exact = 1/8
        gamma_exact = 7/4
        
    elif dim == "3D":
        lattice = "SC"
        NN = 6
        NT = 200
        temperatures = np.linspace(1.5, 7, NT, dtype=np.longdouble)
        L_vals = np.array([4, 6, 7])
        Tc_Exact = 4.51
        
        nu_exact = 0.63
        alpha_exact = 0.11
        beta_exact = 0.33
        gamma_exact = 1.24
    
    beta_vals = 1 / (kB * temperatures)
    
    # Initialization of thermodynamic arrays
    magnetizations = list()
    M = np.zeros((len(L_vals), len(temperatures)), dtype=np.longdouble)
    M2 = np.zeros((len(L_vals), len(temperatures)), dtype=np.longdouble)
    M4 = np.zeros((len(L_vals), len(temperatures)), dtype=np.longdouble)
    mod_M = np.zeros((len(L_vals), len(temperatures)), dtype=np.longdouble)
    
    F = list()
    M_min_F = np.zeros((len(L_vals), len(temperatures)), dtype=np.longdouble)
    min_F = np.zeros((len(L_vals), len(temperatures)), dtype=np.longdouble)
    
    E = np.zeros((len(L_vals), len(temperatures)), dtype=np.longdouble)
    E2 = np.zeros((len(L_vals), len(temperatures)), dtype=np.longdouble)
    mod_E = np.zeros((len(L_vals), len(temperatures)), dtype=np.longdouble)

    mean_C = np.zeros((len(L_vals), len(temperatures)), dtype=np.longdouble)
    mean_S = np.zeros((len(L_vals), len(temperatures)-1), dtype=np.longdouble)
    mean_chi = np.zeros((len(L_vals), len(temperatures)), dtype=np.longdouble) 
    C = np.zeros((len(L_vals), len(temperatures)), dtype=np.longdouble)
    S = np.zeros((len(L_vals), len(temperatures)), dtype=np.longdouble)

    Tc_M_min_F = np.zeros(len(L_vals))
    Tc_mod_M = np.zeros(len(L_vals))
    idx_Tc_mod_M = np.zeros(len(L_vals))
    idx_Tc_min_F = np.zeros(len(L_vals))
    
    for k in range(len(L_vals)):
        L = L_vals[k]
        if dim == "2D":
            N_atm = L ** 2
        elif dim == "3D":
            N_atm = L ** 3
            
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
        
        # Partition function and Helmholtz free energy
        Z = np.zeros(len(temperatures), dtype=np.longdouble)
        Z_M = np.zeros((NM, len(temperatures)), dtype=np.longdouble)
        F.append(np.zeros((NM, len(temperatures)), dtype=np.longdouble))
        
        for q in range(0, NM):
            hits = np.where(JDOS[:, q] > 0)[0]

            for i in range(0, len(hits)):
                Z_M[q, :] += JDOS[hits[i], q] * np.exp(- beta_vals * energies[hits[i]])
            
            Z += Z_M[q, :]
            F[k][q, :] = - kB * temperatures * np.log(Z_M[q, :])
        
        # Magnetizations
        for i in range(0, len(temperatures)):
            for q in range(0, NM):
                M[k][i] += magnetizations[k][q] * Z_M[q, i] / Z[i]
                M2[k][i] += (magnetizations[k][q]**2) * Z_M[q, i] / Z[i]
                M4[k][i] += (magnetizations[k][q]**4) * Z_M[q, i] / Z[i]
                mod_M[k][i] += np.abs(magnetizations[k][q]) * Z_M[q, i] / Z[i]

        # Energies
        for i in range(len(temperatures)):
            for j in range(len(energies)):
                E[k][i] += energies[j] * sum(JDOS[j, :]) * np.exp(- beta_vals[i] * energies[j]) / Z[i]
                E2[k][i] += (energies[j]**2) * sum(JDOS[j, :]) * np.exp(- beta_vals[i] * energies[j]) / Z[i]
                mod_E[k][i] += np.abs(energies[j]) * sum(JDOS[j, :]) * np.exp(- beta_vals[i] * energies[j]) / Z[i]

        # Magnetization for F minima
        for i in range(0, len(temperatures)):
            min_F[k][i] = F[k][0, i]
            q_min = 0
            
            for q in range(0, len(magnetizations[k])):
                if F[k][q, i] < min_F[k][i]:
                    min_F[k][i] = F[k][q, i]
                    q_min = q
            
            M_min_F[k][i] = np.abs(magnetizations[k][q_min])
            
        # Mean magnetic susceptability, mean heat capacity and mean entropy
        for i in range(len(temperatures)):
            mean_C[k][i] = (E2[k][i] - E[k][i]**2) * beta_vals[i]
            mean_chi[k][i] = (M2[k][i] - M[k][i]**2) * beta_vals[i]
            
        mean_S[k] = cumtrapz(mean_C[k] / temperatures, x=temperatures, dx=np.abs(temperatures[0] - temperatures[1]))
        
        # Heat capacity and entropy
        F_sd = np.zeros(len(temperatures), dtype=np.longdouble)
        F_fd = np.zeros(len(temperatures), dtype=np.longdouble)
        h = np.abs(temperatures[1] - temperatures[2])
        for i in range(1, len(temperatures) - 1):
            F_sd[i] = (min_F[k][i - 1] - 2 * min_F[k][i] + min_F[k][i + 1]) / h**2
            F_fd[i] = (min_F[k][i + 1] - min_F[k][i - 1]) / (2 * h)
            
        for i in range(len(temperatures)):
            C[k][i] = - temperatures[i] * F_sd[i]
            S[k][i] = - F_fd[i]
   
        # Tc apprxomation
        h = np.abs(temperatures[1] - temperatures[2])
        M_min_F_fd = np.zeros(len(temperatures), dtype=np.longdouble)
        mod_M_fd = np.zeros(len(temperatures), dtype=np.longdouble)
        
        for i in range(1, len(temperatures) - 1):
            M_min_F_fd[i] = (M_min_F[k][i + 1] - M_min_F[k][i - 1]) / (2 * h)
            mod_M_fd[i] = (mod_M[k][i + 1] - mod_M[k][i - 1]) / (2 * h)
        
        Tc_M_min_F[k] = temperatures[np.where(M_min_F_fd == min(M_min_F_fd))[0][0]]
        idx_Tc_min_F[k] = np.where(M_min_F_fd == min(M_min_F_fd))[0][0]
        Tc_mod_M[k] = temperatures[np.where(mod_M_fd == min(mod_M_fd))[0][0]]
        idx_Tc_mod_M[k] = np.where(mod_M_fd == min(mod_M_fd))[0][0] 
        
        # Normalize computations
        magnetizations[k] /= N_atm
        mod_M[k] /= N_atm
        E[k] /= N_atm
        M_min_F[k] /= N_atm
        F[k] /= N_atm
        min_F[k] /= N_atm
        C[k] /= N_atm
        S[k] /= N_atm
        mean_C[k] /= N_atm
        mean_S[k] /= N_atm
        mean_chi[k] /= N_atm

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
        axs[0, 0].set_title("<|M|> as a function of T")
        
        axs[0, 1].plot(temperatures, E[k], '.-')
        axs[0, 1].set_xlabel("T")
        axs[0, 1].set_ylabel("<E>")
        axs[0, 1].set_title("<E> as a function of T")
        
        axs[1, 0].plot(temperatures, M_min_F[k], '.-')
        axs[1, 0].set_xlabel("T")
        axs[1, 0].set_ylabel("M minF")
        axs[1, 0].set_title("Magnetization for F minina as a function of T")
        
    for i in range(0, len(temperatures)):
        axs[1, 1].plot(magnetizations[0], F[0][:, i], '-b', lw=1)
        axs[1, 1].plot(M_min_F[0][i], min_F[0][i], '.b', ms=7.5)
    axs[1, 1].set_xlabel("M")
    axs[1, 1].set_ylabel("F")
    axs[1, 1].set_title("F as a function of M and T | L = " + str(L_vals[0]))
    
    fig, axs = plt.subplots(2, 2)
    for k in range(len(L_vals)):
        axs[0, 0].plot(temperatures, C[k], '.-')
        axs[0, 0].set_xlabel("T")
        axs[0, 0].set_ylabel("C")
        axs[0, 0].set_title("Heat Capacity per spin as a function of T")
        
        axs[1, 0].plot(temperatures, mean_C[k], '.-')
        axs[1, 0].set_xlabel("T")
        axs[1, 0].set_ylabel("<C>")
        axs[1, 0].set_title("Mean Heat Capacity per spin as a function of T")
        
        axs[0, 1].plot(temperatures, S[k], '.-')
        axs[0, 1].set_xlabel("T")
        axs[0, 1].set_ylabel("S")
        axs[0, 1].set_title("Entropy per spin as a function of T")
        
        axs[1, 1].plot(temperatures[1:], mean_S[k], '.-')
        axs[1, 1].set_xlabel("T")
        axs[1, 1].set_ylabel("<S>")
        axs[1, 1].set_title("Mean Entropy per spin as a function of T")
    
    plt.figure(4)
    for k in range(len(L_vals)):
        plt.plot(temperatures, mean_chi[k])
    
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
    
    
    U = np.zeros((len(L_vals), len(temperatures)), dtype=np.longdouble)
    U = 1 - (M4 / (3 * M2**2))
    plt.figure(6)
    for i in range(0, len(L_vals)):
        plt.plot(temperatures, U[i, :], '.-')

    plt.xlabel("T")
    plt.ylabel("U(T, L)")
    plt.title("Binber Cumulant as a function of L and T")
    
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
    
    # NU - mean value
    
    Tc = np.abs(Tc_mod_M - Tc_Exact)
    
    plt.figure(9)
    plt.plot(np.log(L_vals), np.log(Tc), 'ok')
    
    a = np.polyfit(np.log(L_vals), np.log(Tc), 1)
    nu = - a[0]
    print()
    print("nu: {:.3f}; exact: {:.3f}".format(nu, nu_exact))
    
    plt.plot(np.arange(1, 5, 0.1), a[0] * np.arange(1, 5, 0.1) + a[1], '-b')

    # BETA - mean value
    
    mod_M_Tc = np.zeros(len(L_vals), dtype=np.longdouble)
    for i in range(len(L_vals)):
        mod_M_Tc[i] = mod_M[i, int(idx_Tc_mod_M[i])]
    
    plt.figure(10)
    plt.plot(np.log(L_vals), np.log(mod_M_Tc), '*r')
    
    a = np.polyfit(np.log(L_vals), np.log(mod_M_Tc), 1) 
    beta = a[0]
    print()
    print("beta: {:.3f}; exact: {:.3f}".format( - beta * nu, beta_exact))
    
    plt.plot(np.arange(1, 5, 0.1), a[0] * np.arange(1, 5, 0.1) + a[1], '-b')
    
    # ALPHA - mean value
    
    mean_C_Tc = np.zeros(len(L_vals), dtype=np.longdouble)
    for i in range(len(L_vals)):
        mean_C_Tc[i] = mean_C[i, int(idx_Tc_mod_M[i])]

    plt.figure(11)
    plt.plot(np.log(L_vals), np.log(mean_C_Tc), '*r')
    
    a = np.polyfit(np.log(L_vals), np.log(mean_C_Tc), 1)
    alpha = a[0]
    print()
    print("alpha: {:.3f}; exact: {:.3f}".format(alpha * nu, alpha_exact))
    
    plt.plot(np.arange(1, 5, 0.1), a[0] * np.arange(1, 5, 0.1) + a[1], '-b')
    
    # GAMMA - mean value    
    
    mean_chi_Tc = np.zeros(len(L_vals), dtype=np.longdouble)
    for i in range(len(L_vals)):
        mean_chi_Tc[i] = mean_chi[i, int(idx_Tc_mod_M[i])]
    
    plt.figure(12)
    plt.plot(np.log(L_vals), np.log(mean_chi_Tc), '*r')
    
    a = np.polyfit(np.log(L_vals), np.log(mean_chi_Tc), 1)
    gamma = a[0]
    print()
    print("gamma: {:.3f}; exact: {:.3f}".format(gamma * nu, gamma_exact))
    
    plt.plot(np.arange(1, 5, 0.1), a[0] * np.arange(1, 5, 0.1) + a[1], '-b')
   
    print("Runtime:", time.process_time() - start)
    plt.show()
    
    
if __name__ == '__main__':
    main()
