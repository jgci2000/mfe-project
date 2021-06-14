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
    Tc_Onsager = 2.269
    
    dim = "2D"
    lattice = "SS"
    NN = 4

    NT = 30
    temperatures = np.linspace(0.1, 5, NT)
    beta_vals = 1 / (kB * temperatures)
    
    L_vals = np.array([4, 8, 16])
    
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
    mod_E = np.zeros((len(L_vals), len(temperatures)))

    mean_C = np.zeros((len(L_vals), len(temperatures)))
    mean_S = np.zeros((len(L_vals), len(temperatures)-1))
    C = np.zeros((len(L_vals), len(temperatures)))
    S = np.zeros((len(L_vals), len(temperatures)))

    Tc_M_min_F = np.zeros(len(L_vals))
    Tc_mod_M = np.zeros(len(L_vals))
    
    for k in range(len(L_vals)):
        L = L_vals[k]
        N_atm = 1 * L ** 2
        
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
        Z = np.zeros(len(temperatures))
        Z_M = np.zeros((NM, len(temperatures)))
        F.append(np.zeros((NM, len(temperatures))))
        
        for q in range(0, NM):
            hits = np.where(JDOS[:, q] != 0)[0]

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
            
        mean_S[k] = cumtrapz(mean_C[k] / temperatures, x=temperatures, dx=np.abs(temperatures[0] - temperatures[1]))
        
        # Heat capacity and entropy
        F_sd = np.zeros(len(temperatures))
        F_fd = np.zeros(len(temperatures))
        h = np.abs(temperatures[1] - temperatures[2])
        for i in range(1, len(temperatures) - 1):
            F_sd[i] = (min_F[k][i - 1] - 2 * min_F[k][i] + min_F[k][i + 1]) / h**2
            F_fd[i] = (min_F[k][i + 1] - min_F[k][i - 1]) / (2 * h)
            
        for i in range(len(temperatures)):
            C[k][i] = - temperatures[i] * F_sd[i]
            S[k][i] = - F_fd[i]
   
        # Tc apprxomation
        h = np.abs(temperatures[1] - temperatures[2])
        M_min_F_fd = np.zeros(len(temperatures))
        mod_M_fd = np.zeros(len(temperatures))
        
        for i in range(1, len(temperatures) - 1):
            M_min_F_fd[i] = (M_min_F[k][i + 1] - M_min_F[k][i - 1]) / (2 * h)
            mod_M_fd[i] = (mod_M[k][i + 1] - mod_M[k][i - 1]) / (2 * h)
        
        Tc_M_min_F[k] = temperatures[np.where(M_min_F_fd == min(M_min_F_fd))[0][0]]
        Tc_mod_M[k] = temperatures[np.where(mod_M_fd == min(mod_M_fd))[0][0]]
        
        # Normalize computations
        magnetizations[k] /= N_atm
        mod_M[k] /= N_atm
        M2[k] /= N_atm
        M4[k] /= N_atm
        E[k] /= N_atm
        M_min_F[k] /= N_atm
        F[k] /= N_atm
        min_F[k] /= N_atm
        C[k] /= N_atm
        S[k] /= N_atm
        mean_C[k] /= N_atm
        mean_S[k] /= N_atm

    # Plots
    print()
    for k in range(len(L_vals)):
        print("L{:d} -> Tc_M_min_F: {:.3f}; Tc_mod_M: {:.3f}".format(L_vals[k], Tc_M_min_F[k], Tc_mod_M[k]))

    plt.style.use('seaborn-whitegrid')
    
    fig, axs = plt.subplots(2, 2)
    for k in range(len(L_vals)):
        axs[0, 0].plot(temperatures, mod_M[k], '.-b')
        # axs[0, 0].xlabel("T")
        # axs[0, 0].ylabel("<|M|>")
        axs[0, 0].set_title("<|M|> as a function of T")
        
        axs[0, 1].plot(temperatures, E[k], '.-b')
        # axs[0, 1].xlabel("T")
        # axs[0, 1].ylabel("<E>")
        axs[0, 1].set_title("<E> as a function of T")
        
        axs[1, 0].plot(temperatures, M_min_F[k], '.-b')
        # axs[1, 0].xlabel("T/Tc")
        # axs[1, 0].ylabel("M minF")
        axs[1, 0].set_title("Magnetization for F minina as a function of T")
        
        for i in range(0, len(temperatures)):
            axs[1, 1].plot(magnetizations[k], F[k][:, i], '-b', lw=1)
            axs[1, 1].plot(M_min_F[k][i], min_F[k][i], '.b', ms=7.5)
        # axs[1, 1].xlabel("M")
        # axs[1, 1].ylabel("F")
        axs[1, 1].set_title("F as a function of M and T | L = " + str(L))
    
    fig, axs = plt.subplots(2, 2)
    for k in range(len(L_vals)):
        axs[0, 0].plot(temperatures, C[k], '.-b')
        # axs[0, 0].xlabel("T/Tc")
        # axs[0, 0].ylabel("C")
        axs[0, 0].set_title("Heat Capacity per spin as a function of T | L = " + str(L))
        
        axs[1, 0].plot(temperatures, mean_C[k], '.-b')
        # axs[1, 0].xlabel("T/Tc")
        # axs[1, 0].ylabel("<C>")
        axs[1, 0].set_title("Mean Heat Capacity per spin as a function of T | L = " + str(L))
        
        axs[0, 1].plot(temperatures, S[k], '.-b')
        axs[0, 1].set_title("Entropy per spin as a function of T | L = " + str(L))
        
        axs[1, 1].plot(temperatures[1:], mean_S[k], '.-b')
        axs[1, 1].set_title("Mean Entropy per spin as a function of T | L = " + str(L))
    
    Tc = Tc_mod_M
    plt.figure(3)
    for i in range(0, len(Tc)):
        plt.plot(L_vals[i], Tc[i], 'ok')
    
    plt.plot([0, np.max(L_vals) + 6], [Tc_Onsager, Tc_Onsager], '-r')
    plt.xlim((0, np.max(L_vals) + 6))
    plt.ylim((Tc_Onsager - 0.5, np.max(Tc) + 0.5))
    
    plt.title("Tc vs L")
    black_pnts = mpatches.Patch(color='k', label='Tc')
    plt.legend(handles=[black_pnts])
    red_line = mpatches.Patch(color='r', label='Onsager Tc')
    plt.legend(handles=[red_line, black_pnts])
    
    plt.xlabel("L")
    plt.ylabel("Tc")
    
    plt.figure(4)
    inv_L_vals = 1 / L_vals
    for i in range(0, len(Tc)):
        plt.plot(inv_L_vals[i], Tc[i], 'ok')
    
    plt.plot([- 0.1, np.max(inv_L_vals) + 0.1], [Tc_Onsager, Tc_Onsager], '-r')
    plt.xlim((- 0.1, np.max(inv_L_vals) + 0.1))
    plt.ylim((Tc_Onsager - 0.5, np.max(Tc) + 0.5))
    
    a = np.polyfit(inv_L_vals, Tc, 1)
    
    x = np.linspace(0, 0.30, 100)
    y = a[0] * x + a[1]
    
    plt.plot(x, y, '-b')
    
    plt.title("Tc vs 1/L with linear regression (y={:.3f}x+{:.3f})".format(a[0], a[1]))
    black_pnts = mpatches.Patch(color='black', label='Tc')
    red_line = mpatches.Patch(color='r', label='Onsager Tc')
    blue_line = mpatches.Patch(color='b', label='LinReg')
    plt.legend(handles=[red_line, black_pnts, blue_line])
    
    plt.xlabel("1/L")
    plt.ylabel("Tc")
    
    L_Tc_Onsager = a[0] / (Tc_Onsager - a[1])
    print("L for Tc_Onsager ({:.3f}): {:d}".format(Tc_Onsager, int(L_Tc_Onsager)))
    
    Tc_Lin_Reg = a[1]
    print("Tc by the linear regression from fitted data ->", Tc_Lin_Reg)
    
    U = np.zeros((len(L_vals), len(temperatures)))
    U = 1 - M4 / (3 * M2**2)
    plt.figure(5)
    for i in range(0, len(L_vals)):
        plt.plot(temperatures, U[i, :], '.-')

    plt.xlabel("T")
    plt.ylabel("U(T, L)")
    plt.title("Binber Cumulant as a function of L and T")
    
    idx = np.argwhere(np.diff(np.sign(U[1, :] - U[2, :]))).flatten()
    Tc_Binder = temperatures[idx[len(idx) - 1]]
    print("Tc for the Binder Cumulant ->", Tc_Binder)
    
    print("Runtime:", time.process_time() - start)
    plt.show()
    
    
if __name__ == '__main__':
    main()
