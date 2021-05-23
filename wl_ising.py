# Implementacao Python do WL

import numpy as np

L = 4

# getsystem()

dim = 2
lattice = "SS"
N_atm = L * L
NN = 4

max_E = (1.0 / 2.0) * NN * N_atm
max_M = N_atm

NE = int(1 + (max_E / 2))
NM = N_atm + 1

energies = np.linspace(-max_E,max_E,NE)
magnetizations = np.linspace(-max_M,max_M,NM)

run = 0
f = np.exp(1)
f_final = 0.0
flateness = 0.0

# falta meter os parametros como entradas:

print("No parameters selected, reseting do default.")
run = 0
flatness = 0.9
f_final = 1 + pow(10, - 8)

# string NN_table_file_name = "./neighbour_tables/neighbour_table_" + to_string(dim) + "D_" + lattice + 
# "_" + to_string(NN) + "NN_L" + to_string(L) + ".txt";
# string norm_factor_file_name = "./coefficients/coefficients_" + to_string(N_atm) + "d2.txt";
# string save_file = to_string(run) + "_JDOS_WL_Ising_" + to_string(dim) + "D_" + lattice + "_L" + to_string(L) + "_f" + 
# to_string((int) - log10(f_final - 1)) + "_flatness" + to_string((int) (flatness * 100));

#  Initialize vectors and read files

spins_vector = np.zeros(N_atm)
NN_table = np.zeros((N_atm,NN))
norm_factor = np.zeros(NM)

    # read_NN_talbe(NN_table_file_name, NN_table);
    # read_norm_factor(norm_factor_file_name, norm_factor);

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
    # if ((rand_xoshiro256pp() % 2) + 1 == 1) 
    if np.mod(np.random.rand(1),2) == 0:
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

# idx_E_config = energies[int(E_config)]
idx_M_config = magnetizations[int(M_config)]

# Implementar timing (...)

while f>f_final:

    for idx in range(N_atm):

        flip_idx = int(np.random.rand(1)%N_atm)

        delta_E=0
        for a in range(NN):
            delta_E += - spins_vector[flip_idx] * spins_vector[int(NN_table[flip_idx][a])]

        new_E_config = E_config - 2 * delta_E
        new_M_config  = M_config - 2 * spins_vector[flip_idx]
        new_idx_E_config = energies[int(new_E_config)]
        new_idx_M_config = magnetizations[int(new_M_config)]

        ratio = np.exp(ln_JDOS[idx_E_config][idx_M_config] - ln_JDOS[new_idx_E_config][new_idx_M_config])

        # if (ratio >= 1 || ((ld) rand_xoshiro256pp() / (ld) UINT64_MAX) < ratio)
        if (ratio >= 1 or np.random.rand() < ratio):
            spins_vector[flip_idx] = - spins_vector[flip_idx]
            
            E_config = new_E_config
            idx_E_config = new_idx_E_config
            M_config = new_M_config
            idx_M_config = new_idx_M_config

        hist[idx_E_config][idx_M_config]+=1
        ln_JDOS[idx_E_config][idx_M_config] += np.log(f)

        mc_sweep+=1

        if np.mod(mc_sweep,10000)==0:

            avg_h = average_hist(hist, NE * NM)
            min_h = min_hist(hist, NE * NM)

            if min_h >= avg_h*flateness:

                # timing


                # output


                f=np.sqrt(f)
                mc_sweep=0

                for i in range(NE):
                    for j in range(NM):
                        hist[i][j]=0


## Normalizacao (do matlab):





## output final:

