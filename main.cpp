#include <boost/program_options.hpp>
#include <string>

namespace po = boost::program_options;

void parse_input(int argc, char** argv, std::string image_file_name, int* new_width, int* new_height) {
    // parse command-line arguments

    po::options_description desc("Allowed options");
    desc.add_options()(
        "image_file_name,i", po::value<>(&image_file_name)->required(),
        "path to input file")("w, w", po::value<int>(&new_width)->required(), "new width")
        ("h, h", po::value<int>(&new_height)->required(), "new height");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

}

int main(int argc, char** argv)
{
   std::string image_file_name;
   int new_width;
   int new_height;

   return 0;
}