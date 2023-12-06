#include "driver.hpp"
#include "scexec.hpp"
#include "image.hpp"
#include "matrix.hpp"
#include "timing.hpp"
#include <chrono>
#include <iostream>
using namespace std;

typedef Matrix (*energy_func_t)(timing_t& , Image&);
typedef Matrix (*min_cost_func_t)(timing_t& , Matrix&);

void print_config(energy_func_t, min_cost_func_t, bool);
void print_timing(timing_t timing);

void rough_pipeline(timing_t& timing, Image& image, Matrix& energy_mat, min_cost_func_t min_cost_func) {
    Matrix min_cost_mat = min_cost_func(timing, energy_mat);
    int *seam = find_seam(min_cost_mat);
    energy_mat.remove_seam(seam);
    image.remove_seam(seam);
    delete seam;
}

void smooth_pipeline(timing_t& timing, Image& image, energy_func_t energy_func, min_cost_func_t min_cost_func) {
    Matrix energy_mat = energy_func(timing, image);
    Matrix min_cost_mat = min_cost_func(timing, energy_mat);
    int *seam = find_seam(min_cost_mat);;
    image.remove_seam(seam);
    delete seam;
}

void run_seam_carver(
    Image& image, 
    int new_height,
    int new_width,
    int energy_func_type,
    int min_cost_func_type,
    bool smooth
) {
    energy_func_t energy_func;
    if (energy_func_type == 0) {
        energy_func = compute_energy_mat_seq;
    } else if (energy_func_type == 1) {
        energy_func = compute_energy_mat_par;
    } else if (energy_func_type == 2) {
        energy_func = compute_energy_mat_cuda;
    }

    min_cost_func_t min_cost_func;
    if (min_cost_func_type == 0) {
        min_cost_func = compute_min_cost_mat_seq;
    } else if (min_cost_func_type == 1) {
        min_cost_func = compute_min_cost_mat_par_row;
    } else if (min_cost_func_type == 2) {
        min_cost_func = compute_min_cost_mat_cuda;
    }

    int pipeline_type = smooth ? 1 : 0;

    if (smooth && energy_func_type == 2 && min_cost_func_type == 2) {
        pipeline_type = 2; // cuda without unnecessary copying
    }

    print_config(energy_func, min_cost_func, smooth);

    timing_t timing = {0, 0, 0};
    auto start = TIME_NOW;

    if (pipeline_type == 0) {
        // rough
        Matrix energy_mat = energy_func(timing, image);;
        while (image.width > new_width) {
            rough_pipeline(timing, image, energy_mat, min_cost_func);
        }
        image.transpose();
        energy_mat.transpose();
        while (image.width > new_height) {
            rough_pipeline(timing, image, energy_mat, min_cost_func);
        }
        image.transpose();
    } else if (pipeline_type == 1) {
        // smooth
        while (image.width > new_width) {
            smooth_pipeline(timing, image, energy_func, min_cost_func);
        }
        image.transpose();
        while (image.width > new_height) {
            smooth_pipeline(timing, image, energy_func, min_cost_func);
        }
        image.transpose();
    } else {
        // cuda
        compute_min_cost_mat_direct_cuda(timing, image, new_width);
        image.transpose();
        compute_min_cost_mat_direct_cuda(timing, image, new_height);
        image.transpose();
    }

    timing.total_time = TIME_SINCE(start);
    print_timing(timing);
}

void print_config(energy_func_t energy_func, min_cost_func_t min_cost_func, bool smooth) {
    cout << "------------------CONFIG------------------" << endl;

    if (smooth && energy_func == compute_energy_mat_cuda && min_cost_func == compute_min_cost_mat_cuda) {
        cout << "cuda_end_to_end_func: compute_min_cost_mat_direct_cuda" << endl;
    } else {
        cout << "energy_func: ";
        if (energy_func == compute_energy_mat_seq) {
            cout << "compute_energy_mat_seq" << endl;
        } else if (energy_func == compute_energy_mat_par) {
            cout << "compute_energy_mat_par" << endl;
        } else if (energy_func == compute_energy_mat_cuda) {
            cout << "compute_energy_mat_cuda" << endl;
        }

        cout << "min_cost_func: ";
        if (min_cost_func == compute_min_cost_mat_seq) {
            cout << "compute_min_cost_mat_seq" << endl;
        } else if (min_cost_func == compute_min_cost_mat_par_row) {
            cout << "compute_min_cost_mat_par_row" << endl;
        } else if (min_cost_func == compute_min_cost_mat_cuda) {
            cout << "compute_min_cost_mat_cuda" << endl;
        }

        cout << "smooth: " << (smooth ? "yes" : "no") << endl;
    }

    cout << "------------------------------------------" << endl;
}

void print_timing(timing_t timing) {
    cerr << timing.energy_time << ", " << timing.min_cost_time << ", " << timing.total_time << endl;
}
