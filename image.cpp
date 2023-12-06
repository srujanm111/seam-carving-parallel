#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image.hpp"

#include <cmath>
#include <iostream>
#include <string>

Image::Image(int height, int width) : height{height}, width{width} {
    image = allocate_image(height, width);
}

Image::Image(std::string* image_file_name, bool* success) {
    int channels; 
    unsigned char *stbi_image = stbi_load(image_file_name->c_str(), &width, &height, &channels, 0);

    if (stbi_image != NULL) {
        *success = true;
        image = allocate_image(height, width);

        int k = 0;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                for (int c = 0; c < NUM_CHANNELS; ++c) {
                    if (c < NUM_CHANNELS) {
                        set_pixel(i, j, c, stbi_image[k]);
                    }
                    ++k;
                }
            }
        }
    }
}

Image::~Image() {
    delete_image(image);
}

float** Image::allocate_image(int height, int width) {
    float** allocated_image = new float*[height];
    for (int i = 0; i < height; i++) {
        allocated_image[i] = new float[width * 3];
    }

    return allocated_image;
}

void Image::delete_image(float** delete_image) {
    for (int i = 0; i < height; ++i) {
        delete delete_image[i];
    }

    delete delete_image;
}

float Image::get_pixel(int x, int y, int c) {
    return image[x][y * NUM_CHANNELS + c];
}

void Image::set_pixel(int x, int y, int c, float value) {
    image[x][y * NUM_CHANNELS + c] = value;
}

void Image::remove_seam(int* seam) {
    float** new_image = allocate_image(height, width - 1);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int c = 0; c < NUM_CHANNELS; ++c) {
                if (j < seam[i]) {
                    new_image[i][j * NUM_CHANNELS + c] = get_pixel(i, j, c);
                } else if (j > seam[i]) {
                    new_image[i][(j - 1) * NUM_CHANNELS + c] = get_pixel(i, j, c);
                }
            }
        }
    }

    delete_image(image);
    image = new_image;

    width -= 1;
}

void Image::transpose() {
    float** new_image = allocate_image(width, height);

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int c = 0; c < NUM_CHANNELS; ++c) {
                new_image[i][j * NUM_CHANNELS + c] = get_pixel(j, i, c);
            }
        }
    }

    delete_image(image);
    image = new_image;

    int temp = width;
    width = height;
    height = temp;
}

void Image::output_image(std::string* image_file_output) {
    unsigned char *stbi_image = new unsigned char[width * height * NUM_CHANNELS];

    int k = 0;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int c = 0; c < NUM_CHANNELS; ++c) {
                stbi_image[k++] = (unsigned char) get_pixel(i, j, c);
            }
        }
    }

    if (!stbi_write_jpg(image_file_output->c_str(), width, height, NUM_CHANNELS, stbi_image, 100)) {
        std::cout << "Unable to output image." << std::endl;
    }

    delete stbi_image;
}