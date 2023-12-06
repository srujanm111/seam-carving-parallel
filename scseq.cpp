#include <cmath>
#include "image.hpp"
#include "matrix.hpp"

float dual_gradient_energy_seq(Image& image, int x, int y) {
    int x_up = x == 0 ? image.height - 1 : x - 1;
    int x_down = x == image.height - 1 ? 0 : x + 1;
    int y_left = y == 0 ? image.width - 1 : y - 1;
    int y_right = y == image.width - 1 ? 0 : y + 1;

    float x_gradient = sqrt(
        pow(image.get_pixel(x_up, y, 0) - image.get_pixel(x_down, y, 0), 2) +
        pow(image.get_pixel(x_up, y, 1) - image.get_pixel(x_down, y, 1), 2) +
        pow(image.get_pixel(x_up, y, 2) - image.get_pixel(x_down, y, 2), 2)
    );

    float y_gradient = sqrt(
        pow(image.get_pixel(x, y_left, 0) - image.get_pixel(x, y_right, 0), 2) +
        pow(image.get_pixel(x, y_left, 1) - image.get_pixel(x, y_right, 1), 2) +
        pow(image.get_pixel(x, y_left, 2) - image.get_pixel(x, y_right, 2), 2)
    );

    return x_gradient + y_gradient + 1.0;
}

Matrix compute_energy_mat_seq(Image& image) {
    Matrix energy_mat(image.height, image.width);
    for (int i = 0; i < image.height; i++) {
        for (int j = 0; j < image.width; j++) {
            energy_mat.set(i, j, dual_gradient_energy_seq(image, i, j));
        }
    }
    return energy_mat;
}

Matrix compute_min_cost_mat_seq(Matrix& energies) {
    int height = energies.height;
    int width = energies.width;
    Matrix min_cost_mat(height, width);
    // first row is the same as the first row of the energy matrix
    for (int i = 0; i < width; i++) {
        min_cost_mat.set(0, i, energies.get(0, i));
    }
    // for each row, compute the minimum cost of each pixel
    for (int i = 1; i < height; i++) {
        for (int j = 0; j < width; j++) {
            min_cost_mat.set(i, j, 0);
            for (int k = j - 1; k <= j + 1; k++) {
                if (k < 0 || k >= width) {
                    continue;
                }
                float energy = energies.get(i, j) + min_cost_mat.get(i - 1, k);
                if (min_cost_mat.get(i, j) == 0 || energy < min_cost_mat.get(i, j)) {
                    min_cost_mat.set(i, j, energy);
                }
            }
        }
    }

    return min_cost_mat;
}
