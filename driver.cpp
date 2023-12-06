#include "driver.hpp"
#include "scexec.hpp"
#include "image.hpp"
#include "matrix.hpp"

#include <iostream>
#include <string>
using namespace std;

typedef Matrix (*energy_func_t)(Image&);
typedef Matrix (*min_cost_func_t)(Matrix&);
typedef Matrix (*cuda_min_cost_func_t)(Image&);

void print_config(energy_func_t, min_cost_func_t, cuda_min_cost_func_t, bool);

void rough_pipeline(Image& image, Matrix& energy_mat, min_cost_func_t min_cost_func) {
    Matrix min_cost_mat = min_cost_func(energy_mat);
    int *seam = find_seam(min_cost_mat);
    energy_mat.remove_seam(seam);
    image.remove_seam(seam);
    delete seam;
}

void smooth_pipeline(Image& image, energy_func_t energy_func, min_cost_func_t min_cost_func) {
    Matrix energy_mat = energy_func(image);
    Matrix min_cost_mat = min_cost_func(energy_mat);
    int *seam = find_seam(min_cost_mat);;
    image.remove_seam(seam);
    delete seam;
}

void cuda_smooth_pipeline(Image& image, cuda_min_cost_func_t cuda_min_cost_func) {
    Matrix min_cost_mat = cuda_min_cost_func(image);
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

    std::string output_energy = "energy_mat.png";

    if (pipeline_type == 0) {
        // rough
        Matrix energy_mat = energy_func(image);
        energy_mat.output_image(&output_energy);
        while (image.width > new_width) {
            rough_pipeline(image, energy_mat, min_cost_func);
        }
        image.transpose();
        energy_mat.transpose();
        while (image.width > new_height) {
            rough_pipeline(image, energy_mat, min_cost_func);
        }
        image.transpose();
    } else if (pipeline_type == 1) {
        // smooth
        while (image.width > new_width) {
            smooth_pipeline(image, energy_func, min_cost_func);
        }
        image.transpose();
        while (image.width > new_height) {
            smooth_pipeline(image, energy_func, min_cost_func);
        }
        image.transpose();
    } else {
        // cuda smooth
        while (image.width > new_width) {
            cuda_smooth_pipeline(image, cuda_min_cost_func);
        }
        image.transpose();
        while (image.width > new_height) {
            cuda_smooth_pipeline(image, cuda_min_cost_func);
        }
        image.transpose();
    }
}

void print_config(energy_func_t energy_func, min_cost_func_t min_cost_func, cuda_min_cost_func_t cuda_min_cost_func, bool smooth) {
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
    
    cout << "cuda_min_cost_func: ";
    if (cuda_min_cost_func == compute_min_cost_mat_direct_cuda) {
        cout << "compute_min_cost_mat_direct_cuda" << endl;
    }
    // else if (cuda_min_cost_func == compute_min_cost_mat_direct_cuda_tex) {
    //     cout << "compute_min_cost_mat_direct_cuda_tex" << endl;
    // }

    cout << "smooth: " << smooth << endl;
}
