function M = comp_magnetization2D(spins, L)
% Magnetization of a 2D Ising lattice.

M = 0;

for i = 1:L
    for j = 1:L
        M = M + spins(i, j);
    end
end

end

