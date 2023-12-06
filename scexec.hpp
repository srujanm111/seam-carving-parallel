#ifndef scexec_hpp
#define scexec_hpp

#include "image.hpp"
#include "matrix.hpp"
#include "timing.hpp"

Matrix compute_energy_mat_seq(timing_t& timing, Image& image);

Matrix compute_energy_mat_par(timing_t& timing, Image& image);

Matrix compute_energy_mat_cuda(timing_t& timing, Image& image);

Matrix compute_min_cost_mat_seq(timing_t& timing, Matrix& energies);

Matrix compute_min_cost_mat_par_row(timing_t& timing, Matrix& energies);

Matrix compute_min_cost_mat_cuda(timing_t& timing, Matrix& energies);

void compute_min_cost_mat_direct_cuda(timing_t& timing, Image& image, int new_width);

int *find_seam(Matrix& min_cost_mat);

#endif