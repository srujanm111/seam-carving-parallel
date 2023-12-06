#ifndef image_hpp
#define image_hpp

#include <string>

constexpr int NUM_CHANNELS = 3;

class Image {

    // constructors
    public:
        Image(int height, int width);
        Image(std::string* image_file_name, bool* success);
        ~Image();

    // methods
    public:
        float get_pixel(int x, int y, int c);
        void set_pixel(int x, int y, int c, float value);
        void transpose();
        void remove_seam(int *seam);
        void output_image(std::string* image_file_output);

    // methods
    private:
        float** allocate_image(int height, int width);
        void delete_image(float** image);

    // instance variables
    public:
        float** image;
        int height;
        int width;

};

#endif