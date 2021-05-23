function spins = rand_spins2D(L)
% Generates a L*L matrix of random spins.

spins = zeros(L);

for i = 1:L
    for j = 1:L
        spins(i, j) = sign(2 * rand() - 1);
    end
end

end