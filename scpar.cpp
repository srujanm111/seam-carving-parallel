#include <cmath>
#include "scexec.hpp"
#include "image.hpp"
#include "matrix.hpp"
#include "timing.hpp"

#include <pthread.h>
#include <sys/sysinfo.h>
#include <atomic>

using namespace std;

chrono::_V2::steady_clock::time_point time_start_e, time_end_e;
chrono::_V2::steady_clock::time_point time_start_m, time_end_m;

constexpr int PROCESSOR_THREAD_FACTOR = 1;

static cpu_set_t cpuset;

struct compute_energy_args {
    int start;
    int end;
    Image& image;
    Matrix& energy_mat;
    pthread_barrier_t& time_barrier;
};

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

void *compute_energy_worker_thread(void* arguments) {
    compute_energy_args *args = (struct compute_energy_args *) arguments;

    pthread_barrier_wait(&args->time_barrier);

    if (time_start_e == chrono::steady_clock::time_point()) {
        time_start_e = chrono::steady_clock::now();
    }

    int out_of_bounds = args->image.width * args->image.height;

    for (int i = args->start; i < args->end && i < out_of_bounds; ++i) {
        int x = i / args->image.width;
        int y = i % args->image.width;
        args->energy_mat.set(x, y, dual_gradient_energy_par(args->image, x, y));
    }

    pthread_barrier_wait(&args->time_barrier);

    time_end_e = chrono::steady_clock::now();
    delete args;
}

Matrix compute_energy_mat_par(timing_t& timing, Image& image) {
    Matrix energy_mat(image.height, image.width);

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    int num_processors = get_nprocs();
    int num_threads = num_processors * PROCESSOR_THREAD_FACTOR;
    pthread_t pthreads[num_threads];

    pthread_barrier_t time_barrier;
    pthread_barrierattr_t barrier_attr;
    pthread_barrier_init(&time_barrier, &barrier_attr, num_threads);

    int work_per_thread = (image.height * image.width) / num_threads;
    int extra_work = (image.height * image.width) % num_threads;
    int last_end = 0;

    for (int i = 0; i < num_threads; ++i) {
        // set affinity
        CPU_ZERO(&cpuset);
        CPU_SET(i % num_processors, &cpuset);

        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);

        // add leftover work
        int extra = 0;
        if (extra_work > 0) {
            ++extra;
            --extra_work;
        }

        compute_energy_args* args = new compute_energy_args({last_end, last_end + work_per_thread + extra_work, image, energy_mat, time_barrier});
        pthread_create(&pthreads[i], &attr, compute_energy_worker_thread, args);

        last_end += work_per_thread + extra_work;
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(pthreads[i], NULL);
    }

    chrono::duration<float> time_span = chrono::duration_cast<chrono::duration<double>>(time_end_e - time_start_e);
    timing.energy_time += time_span.count() / 1000;

    return energy_mat;
}

struct compute_min_cost_args {
    int start;
    int end;
    Matrix& energies;
    Matrix& min_cost_mat;
    pthread_barrier_t& row_barrier;
};

void *compute_min_cost_worker_thread(void* arguments) {
    compute_min_cost_args *args = (struct compute_min_cost_args *) arguments;

    pthread_barrier_wait(&args->row_barrier);
    if (time_start_m == chrono::steady_clock::time_point()) {
        time_start_m = chrono::steady_clock::now();
    }
    
    for (int i = 0; i < args->energies.height; ++i) {
        for (int j = args->start; j < args->end & j < args->energies.width; ++j) {
            if (i == 0) {
                args->min_cost_mat.set(0, j, args->energies.get(0, j));
            } else {
                args->min_cost_mat.set(i, j, 0);
                for (int k = j - 1; k <= j + 1; k++) {
                    if (k < 0 || k >= args->energies.width) {
                        continue;
                    }
                    float energy = args->energies.get(i, j) + args->min_cost_mat.get(i - 1, k);
                    if (args->min_cost_mat.get(i, j) == 0 || energy < args->min_cost_mat.get(i, j)) {
                        args->min_cost_mat.set(i, j, energy);
                    }
                }
            }
        }
        pthread_barrier_wait(&args->row_barrier);
    }

    pthread_barrier_wait(&args->row_barrier);
    time_end_m = chrono::steady_clock::now();

    delete args;
}

Matrix compute_min_cost_mat_par_row(timing_t& timing, Matrix& energies) {
    int height = energies.height;
    int width = energies.width;
    Matrix min_cost_mat(height, width);

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    int num_processors = get_nprocs();
    int num_threads = num_processors * PROCESSOR_THREAD_FACTOR;
    pthread_t pthreads[num_threads];

    pthread_barrier_t row_barrier;
    pthread_barrierattr_t barrier_attr;
    pthread_barrier_init(&row_barrier, &barrier_attr, num_threads);

    int work_per_thread = width / num_threads;
    int extra_work = width % num_threads;
    int last_end = 0;

    for (int i = 0; i < num_threads; ++i) {
        // set affinity
        CPU_ZERO(&cpuset);
        CPU_SET(i % num_processors, &cpuset);

        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);

        // add leftover work
        int extra = 0;
        if (extra_work > 0) {
            ++extra;
            --extra_work;
        }

        compute_min_cost_args* args = new compute_min_cost_args({last_end, last_end + work_per_thread + extra_work, energies, min_cost_mat, row_barrier});
        pthread_create(&pthreads[i], &attr, compute_min_cost_worker_thread, args);

        last_end += work_per_thread + extra_work;
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(pthreads[i], NULL);
    }

    chrono::duration<float> time_span = chrono::duration_cast<chrono::duration<double>>(time_end_m - time_start_m);
    timing.min_cost_time += time_span.count() / 1000;

    return min_cost_mat;
}

Matrix compute_min_cost_mat_par_tri(Matrix& energies) {
    // FUTURE: implement triangular parallelism
}
