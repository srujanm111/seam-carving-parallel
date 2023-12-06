#include "scexec.hpp"
#include "matrix.hpp"

int *find_seam(Matrix& min_cost_mat) {
    int height = min_cost_mat.height;
    int width = min_cost_mat.width;
    int *seam = new int[height];
    int min_index = 0;
    for (int i = 0; i < width; i++) {
        if (min_cost_mat.get(height - 1, i) < min_cost_mat.get(height - 1, min_index)) {
            min_index = i;
        }
    }
    seam[height - 1] = min_index;
    for (int i = height - 2; i >= 0; i--) {
        int min_index = seam[i + 1];
        for (int j = min_index - 1; j <= min_index + 1; j++) {
            if (j < 0 || j >= width) {
                continue;
            }
            if (min_cost_mat.get(i, j) < min_cost_mat.get(i, min_index)) {
                min_index = j;
            }
        }
        seam[i] = min_index;
    }
    return seam;
}
