#ifndef driver_hpp
#define driver_hpp

#include "image.hpp"

void run_seam_carver(
    Image& image, 
    int new_height,
    int new_width,
    int energy_func_type,
    int min_cost_func_type,
    bool smooth
);

#endif
