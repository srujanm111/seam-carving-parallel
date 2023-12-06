#ifndef main_hpp
#define main_hpp

#include <string>

void parse_input(int argc, char** argv, std::string* image_file_name, std::string* image_file_output, int* new_width, int* new_height, int* energy_func_type, int* min_cost_type, bool* smooth);

#endif