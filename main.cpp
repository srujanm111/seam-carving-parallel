#define STB_IMAGE_IMPLEMENTATION

#include "image.hpp"
#include "driver.hpp"
#include <boost/program_options.hpp>
#include <iostream>
#include <string>

namespace po = boost::program_options;

void parse_input(int argc, char** argv, std::string* image_file_input, std::string* image_file_output, int* new_width, int* new_height, int* energy_func_type, int* min_cost_type, bool* smooth) {
    po::options_description desc("Allowed options");
    desc.add_options()
    ("image_file_input,i", po::value<std::string>(image_file_input)->required(),"path to input file")
    ("image_file_output,o", po::value<std::string>(image_file_output)->required(),"path to output file")
    ("width,w", po::value<int>(new_width)->required(), "new width")
    ("height,h", po::value<int>(new_height)->required(), "new height")
    ("energy_func_type,e", po::value<int>(energy_func_type), "todo")
    ("min_cost_type,m", po::value<int>(min_cost_type), "todo")
    ("smooth,s", "todo");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    *smooth = vm.count("smooth");
}

int main(int argc, char** argv) {
    std::string image_file_input;
    std::string image_file_output;
    int new_width;
    int new_height;

    int energy_func_type;
    int min_func_type;
    bool smooth;

    int width;
    int height;

    // parse command-line arguments
    parse_input(argc, argv, &image_file_input, &image_file_output, &new_width, &new_height, &energy_func_type, &min_func_type, &smooth);

    // process image file
    bool success;
    Image image(&image_file_input, &success);

    if (!success) {
        // error msg
        std::cout << "Unable to read provided image path." << std::endl;
        return 1;
    }

    run_seam_carver(
        image, 
        new_height,
        new_width,
        energy_func_type,
        min_func_type,
        smooth
    );

    image.output_image(&image_file_output);

    return 0;
}