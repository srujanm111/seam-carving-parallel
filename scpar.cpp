#include <cmath>
#include "scexec.hpp"
#include "image.hpp"
#include "matrix.hpp"

#include <pthread.h>
#include <sys/sysinfo.h>

constexpr int PROCESSOR_THREAD_FACTOR = 2;

static cpu_set_t cpuset;

struct compute_energy_args {
    int start;
    int end;
    Image& image;
    Matrix& matrix;
};

void *compute_energy_worker_thread(void* update_addr) {
    
}

float dual_gradient_energy_par(Image& image, int x, int y) {
    int x_left = x == 0 ? image.height - 1 : x - 1;
    int x_right = x == image.height - 1 ? 0 : x + 1;
    int y_up = y == 0 ? image.width - 1 : y - 1;
    int y_down = y == image.width - 1 ? 0 : y + 1;

    float x_gradient = sqrt(
        pow(image.get_pixel(x_left, y, 0) - image.get_pixel(x_right, y, 0), 2) +
        pow(image.get_pixel(x_left, y, 1) - image.get_pixel(x_right, y, 1), 2) +
        pow(image.get_pixel(x_left, y, 2) - image.get_pixel(x_right, y, 2), 2)
    );

    float y_gradient = sqrt(
        pow(image.get_pixel(x, y_up, 0) - image.get_pixel(x, y_down, 0), 2) +
        pow(image.get_pixel(x, y_up, 1) - image.get_pixel(x, y_down, 1), 2) +
        pow(image.get_pixel(x, y_up, 2) - image.get_pixel(x, y_down, 2), 2)
    );

    return x_gradient + y_gradient + 1.0;
}

Matrix compute_energy_mat_par(Image& image) {
    Matrix energy_mat(image.height, image.width);

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    int num_processors = get_nprocs();
    int num_threads = num_processors * PROCESSOR_THREAD_FACTOR;
    pthread_t pthreads[num_threads];

    int work_per_thread = (image.height * image.width) / num_threads;
    int extra_work = (image.height * image.width) % num_threads;
    int last_end = 0;

    for (int i = 0; i < num_threads; ++i) {
        // set affinity
        CPU_ZERO(&cpuset);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);

        // add leftover work
        int extra = 0;
        if (extra_work > 0) {
            ++extra;
            --extra_work;
        }

        // compute_energy_args* args = new compute_energy_args({last_end, last_end + extra_work, image, matrix});
        // last_end += extra_work;

        // pthread_create(&pthreads[i], &attr, dual_gradient_energy_par, );
    }
}

Matrix compute_min_cost_mat_par_row(Matrix& energies) {

}

Matrix compute_min_cost_mat_par_tri(Matrix& energies) {

}
