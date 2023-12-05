float ***compute_energy_mat_seq(float ***image, int length, int width);

float ***compute_energy_mat_par(float ***image, int length, int width);

float ***compute_energy_mat_cuda(float ***image, int length, int width);

float ***compute_min_cost_mat_seq(float ***energies, int length, int width);

float ***compute_min_cost_mat_par(float ***energies, int length, int width);

float ***compute_min_cost_mat_cuda(float ***energies, int length, int width);