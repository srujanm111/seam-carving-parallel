#ifndef TIMING_HPP
#define TIMING_HPP

#include <chrono>

#define TIME_NOW std::chrono::high_resolution_clock::now()

#define TIME_SINCE(start) std::chrono::duration_cast<std::chrono::duration<float>>(TIME_NOW - start).count()

typedef struct {
    float energy_time;
    float min_cost_time;
    float total_time;
} timing_t;

#endif