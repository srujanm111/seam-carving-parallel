#ifndef scexec_hpp
#define scexec_hpp

#include "image.hpp"
#include "matrix.hpp"

Matrix compute_energy_mat_seq(Image& image);

Matrix compute_energy_mat_par(Image& image);

Matrix compute_energy_mat_cuda(Image& image);

// Matrix compute_energy_mat_cuda_tex(Image& image);

Matrix compute_min_cost_mat_seq(Matrix& energies);

Matrix compute_min_cost_mat_par_row(Matrix& energies);

Matrix compute_min_cost_mat_par_tri(Matrix& energies);

Matrix compute_min_cost_mat_cuda(Matrix& energies);

Matrix compute_min_cost_mat_direct_cuda(Image& image);

// Matrix compute_min_cost_mat_direct_cuda_tex(Image& image);

int *find_seam(Matrix& min_cost_mat);

#endif