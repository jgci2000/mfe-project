clear 
close

% Wang Landau sampling for 2D Ising Model (JDOS)
% João Inácio, Aug. 29, 2020

    % Constants
L = 2;              % Number of spins for each side of the lattice
NN = 4;             % Number of neighvour spins
N_SPINS = L * L;     % Number of spins
J = 1;              % Interaction strenght between particles

    % Neighbours Vectors
aux = 1:L;
menos = circshift(aux, [0 -1]);
mais = circshift(aux, [0 +1]);

    % Wang Landau Sampling
f = exp(1);     % Modification factor
f_final = 1 + 1E-8;
flatness = 0.90;       % Flatness -> min(hist) > avg(hist)*p

% All of the possible energies
energies = - (1 / 2) * NN * N_SPINS:4:(1 / 2) * NN * N_SPINS;
NE = N_SPINS + 1;

% All of the possible magnetizations. 
magnetizations = - N_SPINS:2:N_SPINS;
NM = N_SPINS + 1;

% Random configuration
spins = randSpins(L);
E = comp_energy2D(spins, J, L);
M = comp_magnetization2D(spins, L);
idx_E = find(energies == E);
idx_M = find(magnetizations == M);

% Joint Desnsity of States Vector
JDOS = ones(NE, NM);
% Rows -> Energies columns -> Magnetizations
% Working with the ln(g(E)) is better because the values are too high.
lnJDOS = log(JDOS);

% At the end of 10000 MCSweeps check if the histogram is flat.
mc_sweeps = 0;   % Reset the sweeps counter.

% Histogram
hist = zeros(NE, NM);
% Rows -> Energies columns -> Magnetizations

tic
while f > f_final
    % Each for loop is 1 MCSweep
    for ni = 1:N_SPINS
        % Select a random spin
        i = randi([1 L]);
        j = randi([1 L]);
        S = spins(i, j);
        
        % dE = 2 * J * S * sum(neighbours) and dE = ENew - E, so
        sum_nei = spins(mais(i), j) + spins(i, mais(j))...
            + spins(menos(i), j) + spins(i, menos(j));
        new_E = E + 2 * J * S * sum_nei;
        new_M = M - 2 * S;
        idx_new_E = find(energies == new_E);
        idx_new_M = find(magnetizations == new_M);
        
        % Flip the spin
        ratio = exp(lnJDOS(idx_E, idx_M) - lnJDOS(idx_new_E, idx_new_M));
        P = min(ratio, 1);
        
        if P > rand()
            spins(i, j) = -S;
            E = new_E;
            M = new_M;
            idx_E = idx_new_E;
            idx_M = idx_new_M;
        end
        
        % Update the histogram and g(E)
        hist(idx_E, idx_M) = hist(idx_E, idx_M) + 1;
        lnJDOS(idx_E, idx_M) = lnJDOS(idx_E, idx_M) + log(f);
    end
    
    mc_sweeps = mc_sweeps + 1;
    
    % Check the flatness of the histogram each 10000 setps.
    if mod(mc_sweeps, 10000) == 0
        avg_hist = mean(mean(hist(hist > 0)));
        min_hist = min(min(hist(hist > 0)));
        
        if min_hist > avg_hist * flatness
            fprintf("%d: the histogram is flat. Min: %.0f Avg: %.2f f: %.8f\n", mc_sweeps, min_hist, avg_hist, f);
            
            f = f^(1/2);
            mc_sweeps = 0;
            hist = zeros(NE, NM);
        end
    end
end
toc

% Normalize the DOS
% lngE is the normalized Density of States.
lnJDOS(lnJDOS > 0) = lnJDOS(lnJDOS > 0) - lnJDOS(energies == - (1 / 2) * NN * N_SPINS, magnetizations == - N_SPINS) + log(2);
% C = log(NConfig) - sum(sum(lngEM));
% lngEM(lngEM > 0) = lngEM(lngEM > 0) + C;
% Get the actual JDOS
JDOS = zeros(NE, NM);

for i = 1:NE
    for j = 1:NM
        if lnJDOS(i, j) ~= 0
            JDOS(i, j) = exp(lnJDOS(i, j)) / 2;
        end
    end
end

figure(2)
bar3(JDOS)
xlabel("Magnetizations")
ylabel("Energies")
zlabel("Normalized JDOS")




