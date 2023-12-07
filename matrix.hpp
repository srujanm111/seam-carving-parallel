#ifndef matrix_hpp
#define matrix_hpp

#include <string>
#include <iostream>
#include "image.hpp"

class Matrix {
private:
    float **allocate_matrix(int height, int width) {
        float** allocated_matrix = new float*[height];
        for (int i = 0; i < height; i++) {
            allocated_matrix[i] = new float[width];
        }

        return allocated_matrix;
    }

    void delete_matrix(float** delete_matrix) {
        for (int i = 0; i < height; ++i) {
            delete delete_matrix[i];
        }

        delete delete_matrix;
    }

public:
    float **matrix;
    int height;
    int width;

    Matrix(int height, int width) : height{height}, width{width} {
        matrix = allocate_matrix(height, width);
    }

    Matrix(float **matrix, int height, int width) : matrix{matrix}, height{height}, width{width} {}

    ~Matrix() {
        delete_matrix(matrix);
    }

    float get(int x, int y) {
        return matrix[x][y];
    }

    void set(int x, int y, float value) {
        matrix[x][y] = value;
    }

    void transpose() {
        float **transposed_matrix = allocate_matrix(width, height);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                transposed_matrix[i][j] = matrix[j][i];
            }
        }
        delete_matrix(matrix);
        matrix = transposed_matrix;
        int temp = height;
        height = width;
        width = temp;
    }

    void remove_seam(int *seam) {
        float **new_matrix = allocate_matrix(height, width - 1);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width - 1; j++) {
                if (j < seam[i]) {
                    new_matrix[i][j] = matrix[i][j];
                } else {
                    new_matrix[i][j] = matrix[i][j + 1];
                }
            }
        }
        delete_matrix(matrix);
        matrix = new_matrix;
        width--;
    }

    void output_image(std::string *image_file_output) {
        Image image(height, width);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; ++j) {
                image.set_pixel(i, j, 0, matrix[i][j]);
                image.set_pixel(i, j, 1, matrix[i][j]);
                image.set_pixel(i, j, 2, matrix[i][j]);
            }
        }
        image.output_image(image_file_output);
    }
};

#endif