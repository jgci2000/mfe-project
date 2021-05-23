# Implementacao Python do WL

import numpy as np
import time

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

energies=dict.fromkeys(energies_keys)
magnetizations=dict.fromkeys(magnetizations_keys)

idx_temp=0
for x in energies_keys:
    energies[x]=idx_temp
    idx_temp+=1 

idx_temp=0
for x in magnetizations_keys:
    magnetizations[x]=idx_temp
    idx_temp+=1 

run = 0
f = np.exp(1)
flatness = 0.9
f_final = 1 + pow(10, - 8)

console_output = "run: " + str(run) + " | L: " + str(L) + " | f_final: 1+1E" + str(int(np.log10(f_final - 1))) + " | flatness: " + str((int) (flatness * 100)) + " | dim: " + str(dim) + "D | lattice: " + lattice
print(console_output)

#  Initialize vectors and read files

spins_vector = np.zeros(N_atm)

NN_table_file_name = "./neighbour_tables/neighbour_table_" + str(dim) + "D_" + lattice + "_" + str(NN) + "NN_L" + str(L) + ".txt"
norm_factor_file_name = "./coefficients/coefficients_" + str(N_atm) + "d2.txt"

NN_table=np.loadtxt(NN_table_file_name, delimiter=' ')
norm_factor=np.loadtxt(norm_factor_file_name, delimiter=' ')

# print(NN_table)
# print(norm_factor)

ln_JDOS = np.zeros((NE,NM))
JDOS = np.zeros((NE,NM))
hist = np.zeros((NE,NM))

for i in range(NE):
    for j in range(NM):
        JDOS[i][j]=0
        ln_JDOS[i][j]=0
        hist[i][j]=0

mc_sweep=0

for i in range(N_atm):
    if np.random.randint(2) == 0:
        spins_vector[i] = 1
    else: 
        spins_vector[i] = -1

E_config = 0
M_config = 0

for i in range(N_atm):
    for a in range(NN):
        E_config += -spins_vector[i] * spins_vector[int(NN_table[i][a])]
    M_config += spins_vector[i]

E_config /= 2

idx_E_config = energies[E_config]
idx_M_config = magnetizations[M_config]

# Implementar timing
method_start= time.perf_counter()

loop_start= time.perf_counter()

while f>f_final:

    if mc_sweep == 0:
        loop_start = time.perf_counter()

    for idx in range(N_atm):

        flip_idx = np.random.randint(N_atm)

        delta_E=0
        for a in range(NN):
            delta_E += - spins_vector[flip_idx] * spins_vector[int(NN_table[flip_idx][a])]

        new_E_config = E_config - 2 * delta_E
        new_M_config  = M_config - 2 * spins_vector[flip_idx]
        new_idx_E_config = energies[new_E_config]
        new_idx_M_config = magnetizations[new_M_config]

        ratio = np.exp(ln_JDOS[idx_E_config][idx_M_config]- ln_JDOS[new_idx_E_config][new_idx_M_config])
        
        if (ratio >= 1 or np.random.rand(1) < ratio):
            spins_vector[flip_idx] = - spins_vector[flip_idx]
            
            E_config = new_E_config
            idx_E_config = new_idx_E_config
            M_config = new_M_config
            idx_M_config = new_idx_M_config

        hist[idx_E_config][idx_M_config]+=1
        ln_JDOS[idx_E_config][idx_M_config] += np.log(f)
        JDOS[idx_E_config][idx_M_config] += f
        
        mc_sweep+=1

        if np.mod(mc_sweep,10000)==0:

            avg_h = np.average(hist[hist!=0])
            min_h = np.min(hist[hist!=0])
            # print(hist)
            # print(avg_h,min_h)

            if min_h >= avg_h*flatness:

                # timing

                loop_dur=time.perf_counter()-loop_start    
                console_output = "f: 1+1E" + str(np.log10(f - 1)) + "/1+1E" + str(int(np.log10(f_final - 1))) + " | sweeps: " + str(mc_sweep) + " | flat time: " + str(loop_dur) + "s"
                print(console_output)

                f=np.sqrt(f)
                mc_sweep=0

                for i in range(NE):
                    for j in range(NM):
                        hist[i][j]=0


## Normalizacao (do matlab):

# # ln_JDOS[ln_JDOS > 0] = ln_JDOS[ln_JDOS > 0] - ln_JDOS[energies == - (1 / 2) * NN * N_atm][magnetizations == - N_atm] + np.log(2)
# JDOS = np.zeros((NE, NM))

# for i in range(NE):
#     for j in range(NM):
#         if 0< ln_JDOS[i][j] < 0.001:
#             JDOS[i][j] = np.exp(ln_JDOS[i][j])/2

#  output final:
runtime= time.perf_counter()-method_start
print("Runtime: ",runtime,"s")

print("JDOS:\n")
print(JDOS)
