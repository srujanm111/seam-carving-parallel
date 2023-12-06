#ifndef driver_hpp
#define driver_hpp

#include "image.hpp"

typedef struct {
    float energy_time;
    float min_cost_time;
    float total_time;
} timing_t;

void run_seam_carver(
    Image& image, 
    int new_height,
    int new_width,
    int energy_func_type,
    int min_cost_func_type,
    bool smooth
);

#endif
