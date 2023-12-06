#include "driver.hpp"
#include "scexec.hpp"
#include "image.hpp"
#include "matrix.hpp"
#include <chrono>
#include <iostream>
using namespace std;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;

typedef Matrix (*energy_func_t)(Image&);
typedef Matrix (*min_cost_func_t)(Matrix&);
typedef Matrix (*cuda_min_cost_func_t)(Image&);

void print_config(energy_func_t, min_cost_func_t, cuda_min_cost_func_t, bool);
float time_since(std::chrono::time_point<Time> start);
void print_timing(timing_t timing);

void rough_pipeline(timing_t *timing, Image& image, Matrix& energy_mat, min_cost_func_t min_cost_func) {
    auto start = Time::now();
    Matrix min_cost_mat = min_cost_func(energy_mat);
    timing->min_cost_time += time_since(start);
    int *seam = find_seam(min_cost_mat);
    energy_mat.remove_seam(seam);
    image.remove_seam(seam);
    delete seam;
}

void smooth_pipeline(timing_t *timing, Image& image, energy_func_t energy_func, min_cost_func_t min_cost_func) {
    auto start = Time::now();
    Matrix energy_mat = energy_func(image);
    timing->energy_time += time_since(start);
    start = Time::now();
    Matrix min_cost_mat = min_cost_func(energy_mat);
    timing->min_cost_time += time_since(start);
    int *seam = find_seam(min_cost_mat);;
    image.remove_seam(seam);
    delete seam;
}

void cuda_smooth_pipeline(timing_t *timing, Image& image, cuda_min_cost_func_t cuda_min_cost_func) {
    auto start = Time::now();
    Matrix min_cost_mat = cuda_min_cost_func(image);
    timing->min_cost_time += time_since(start);
    int *seam = find_seam(min_cost_mat);
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
    } else if (energy_func_type == 3) {
        // energy_func = compute_energy_mat_cuda_tex;
    }

    min_cost_func_t min_cost_func;
    if (min_cost_func_type == 0) {
        min_cost_func = compute_min_cost_mat_seq;
    } else if (min_cost_func_type == 1) {
        min_cost_func = compute_min_cost_mat_par_row;
    } else if (min_cost_func_type == 2) {
        min_cost_func = compute_min_cost_mat_par_tri;
    } else if (min_cost_func_type == 3) {
        min_cost_func = compute_min_cost_mat_cuda;
    }

    int pipeline_type = smooth ? 1 : 0;

    cuda_min_cost_func_t cuda_min_cost_func = NULL;
    if (smooth && (energy_func_type == 2 || energy_func_type == 3) && min_cost_func_type == 3) {
        // cuda without unnecessary copy
        pipeline_type = 2;
        if (energy_func_type == 2) {
            cuda_min_cost_func = compute_min_cost_mat_direct_cuda;
        } else {
            // cuda_min_cost_func = compute_min_cost_mat_direct_cuda_tex;
        }
    }

    print_config(energy_func, min_cost_func, cuda_min_cost_func, smooth);

    timing_t timing = {0, 0, 0};
    auto start = Time::now();

    if (pipeline_type == 0) {
        // rough
        Matrix energy_mat = energy_func(image);;
        while (image.width > new_width) {
            rough_pipeline(&timing, image, energy_mat, min_cost_func);
        }
        image.transpose();
        energy_mat.transpose();
        while (image.width > new_height) {
            rough_pipeline(&timing, image, energy_mat, min_cost_func);
        }
        image.transpose();
    } else if (pipeline_type == 1) {
        // smooth
        while (image.width > new_width) {
            smooth_pipeline(&timing, image, energy_func, min_cost_func);
        }
        image.transpose();
        while (image.width > new_height) {
            smooth_pipeline(&timing, image, energy_func, min_cost_func);
        }
        image.transpose();
    } else {
        // cuda smooth
        while (image.width > new_width) {
            cuda_smooth_pipeline(&timing, image, cuda_min_cost_func);
        }
        image.transpose();
        while (image.width > new_height) {
            cuda_smooth_pipeline(&timing, image, cuda_min_cost_func);
        }
        image.transpose();
    }

    timing.total_time = time_since(start);
    print_timing(timing);
}

void print_config(energy_func_t energy_func, min_cost_func_t min_cost_func, cuda_min_cost_func_t cuda_min_cost_func, bool smooth) {
    cout << "------------------CONFIG------------------" << endl;

    if (cuda_min_cost_func == compute_min_cost_mat_direct_cuda) {
        cout << "cuda_end_to_end_func: compute_min_cost_mat_direct_cuda" << endl;
        return;
    }
    // else if (cuda_min_cost_func == compute_min_cost_mat_direct_cuda_tex) {
    //     cout << "cuda_end_to_end_func: compute_min_cost_mat_direct_cuda_tex" << endl;
    //     return;
    // }

    cout << "energy_func: ";
    if (energy_func == compute_energy_mat_seq) {
        cout << "compute_energy_mat_seq" << endl;
    } else if (energy_func == compute_energy_mat_par) {
        cout << "compute_energy_mat_par" << endl;
    } else if (energy_func == compute_energy_mat_cuda) {
        cout << "compute_energy_mat_cuda" << endl;
    }
    //  else if (energy_func == compute_energy_mat_cuda_tex) {
    //     cout << "compute_energy_mat_cuda_tex" << endl;
    // }

    cout << "min_cost_func: ";
    if (min_cost_func == compute_min_cost_mat_seq) {
        cout << "compute_min_cost_mat_seq" << endl;
    } else if (min_cost_func == compute_min_cost_mat_par_row) {
        cout << "compute_min_cost_mat_par_row" << endl;
    } else if (min_cost_func == compute_min_cost_mat_par_tri) {
        cout << "compute_min_cost_mat_par_tri" << endl;
    } else if (min_cost_func == compute_min_cost_mat_cuda) {
        cout << "compute_min_cost_mat_cuda" << endl;
    }

    cout << "smooth: " << (smooth ? "yes" : "no") << endl;

    cout << "------------------------------------------" << endl;
}

float time_since(std::chrono::time_point<Time> start) {
    auto end = Time::now();
    fsec fs = end - start;
    return fs.count();
}

void print_timing(timing_t timing) {
    cerr << timing.energy_time << ", " << timing.min_cost_time << ", " << timing.total_time << endl;
}
