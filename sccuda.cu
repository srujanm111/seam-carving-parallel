#include "scexec.hpp"
#include "image.hpp"
#include "matrix.hpp"

#include <iostream>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define BLOCK_DIM 16
#define THREADS_PER_BLOCK 256
#define NUM_COLORS 3

int find_cuda_device() {
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (numDevices > 1) {
        int maxMultiprocessors = 0, maxDevice = 0;
        for (int device=0; device<numDevices; device++) {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, device);
            if (maxMultiprocessors < props.multiProcessorCount) {
                maxMultiprocessors = props.multiProcessorCount;
                maxDevice = device;
            }
        }
        cudaSetDevice(maxDevice);
        return maxDevice;
    }

    return 0;
}

__device__ float* get_pixel(float* image, int width, int x, int y) {
  return image + (y * width + x) * 3;
}

__device__ float x_gradient(float* image, int height, int width, int x, int y) {
  int x_left = x != 0 ? x - 1 : width - 1;
  int x_right = x != width - 1 ? x + 1 : 0;

  float* left = get_pixel(image, width, x_left, y);
  float* right = get_pixel(image, width, x_right, y);

  float r = powf(left[0] - right[0], 2);
  float g = powf(left[1] - right[1], 2);
  float b = powf(left[2] - right[2], 2);

  return sqrtf(r + g + b);
}

__device__ float y_gradient(float* image, int height, int width, int x, int y) {
  int y_above = y != 0 ? y - 1 : height - 1;
  int y_below = y != height - 1 ? y + 1 : 0;

  float* above = get_pixel(image, width, y_above, y);
  float* below = get_pixel(image, width, y_below, y);

  float r = powf(above[0] - below[0], 2);
  float g = powf(above[1] - below[1], 2);
  float b = powf(above[2] - below[2], 2);

  return sqrtf(r + g + b);
}

__global__ void dual_gradient(float* image, int height, int width, float* energies) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < width && y < height) {
    float x_grad = x_gradient(image, height, width, x, y);
    float y_grad = y_gradient(image, height, width, x, y);

    int energy_idx = y * width + x;
    energies[energy_idx]  = x_grad + y_grad + 1.0;
  }
}

__device__ float get_energy(float* energies, int width, int x, int y) {
  int idx = y * width + x;
  return energies[idx];
}

__global__ void min_cost(float* energies, int width, int row, float* min_energies) {
  int y = row;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  printf("%d\n", x);

  if(x < width) {
    float min_energy_above = 0;
    if(y > 0) {
      min_energy_above = get_energy(energies, width, x, y - 1);
      if(x > 0) {
        float left = get_energy(energies, width, x - 1, y - 1);
        min_energy_above = min(min_energy_above, left);
      }
      if(x < width - 1) {
        float right = get_energy(energies, width, x + 1, y - 1);
        min_energy_above = min(min_energy_above, right);
      }
    }

    int idx = y * width + x;
    min_energies[idx] = min_energy_above + get_energy(energies, width, x, y);
  }
}

/**
* Uses dual gradient energy to create an energy matrix of the image pixels
* @param image the image
* @return a 2d float matrix (height, width) representing the min energy of each pixel
*/
__host__ Matrix compute_min_cost_mat_direct_cuda(Image& image) {
  find_cuda_device();

  float height = image.height;
  float width = image.width;
  
  float** image_data = image.image;

  //Send image to GPU
  const int num_points = height * width;

  //Allocate image
  float* image_flat;
  cudaMalloc(&image_flat, num_points * NUM_COLORS * sizeof(float));
  const int pixel_row_size = width * NUM_COLORS * sizeof(float);
  for(int y = 0; y < height; ++y) {
    int img_idx = y * width * NUM_COLORS;
    cudaMemcpy(image_flat + img_idx, image_data[y], pixel_row_size, cudaMemcpyHostToDevice);
  }

  //Allocate energies
  float* energies;
  cudaMalloc(&energies, num_points * sizeof(float));
  cudaMemset(energies, 0, num_points * sizeof(float));

  //Block decomposition
  dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
  dim3 num_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
                  (height + threads_per_block.y - 1) / threads_per_block.y);

  //launch kernel
  dual_gradient<<<num_blocks, threads_per_block>>>(image_flat, height, width, energies);

  //Allocate energies
  float* min_energies;
  cudaMalloc(&min_energies, num_points * sizeof(float));

  //TODO new decomposition
  int energy_blocks = width / THREADS_PER_BLOCK;
  for(int y = 0; y < height; ++y) {  
    min_cost<<<energy_blocks, THREADS_PER_BLOCK>>>(energies, width, y, min_energies);
  }

  float** out_energies = (float**)malloc(height * sizeof(float*));
  const int row_size = width * sizeof(float);
  for(int y = 0; y < height; y++) {
    out_energies[y] = (float*)malloc(width * sizeof(float));
    int offset = y * width;
    cudaMemcpy(out_energies[y], min_energies + offset, row_size, cudaMemcpyDeviceToHost);
  }

  cudaFree(energies);
  cudaFree(min_energies);

  //Should be no need for barriers thanks to final cuda memcpy
  return Matrix(out_energies, height, width);
}

__host__ Matrix compute_energy_mat_cuda(Image& image) {
  find_cuda_device();

  float height = image.height;
  float width = image.width;
  float** image_data = image.image;

  //Send image to GPU
  const int num_points = height * width;

  //Allocate image
  float* image_flat;
  cudaMalloc(&image_flat, num_points * NUM_COLORS * sizeof(float));
  const int pixel_row_size = width * NUM_COLORS * sizeof(float);
  for(int r = 0; r < height; ++r) {
    int img_idx = r * width * NUM_COLORS;
    cudaMemcpy(image_flat + img_idx, image_data[r], pixel_row_size, cudaMemcpyHostToDevice);
  }

  //Allocate energies
  float* energies;
  cudaMalloc(&energies, num_points * sizeof(float));
  cudaMemset(energies, 0, num_points * sizeof(float));

  //Block decomposition
  dim3 block_dims(ceil((double)height / BLOCK_DIM), ceil((double)width / BLOCK_DIM), 1);
  dim3 thread_dims(BLOCK_DIM, BLOCK_DIM, 1);

  //launch kernel
  dual_gradient<<<block_dims, thread_dims>>>(image_flat, height, width, energies);

  float** out_energies = (float**) malloc(height * sizeof(float*));
  const int row_size = width * sizeof(float);
  for(int y = 0; y < height; y++) {
    out_energies[y] = (float*) malloc(width * sizeof(float));
    int offset = y * width;
    cudaMemcpy(out_energies[y], energies + offset, row_size, cudaMemcpyDeviceToHost);
  }

  cudaFree(energies);

  //Should be no need for barriers thanks to final cuda memcpy
  return Matrix(out_energies, height, width);
}

__host__ Matrix compute_min_cost_mat_cuda(Matrix& energies_mat) {
  find_cuda_device();

  float height = energies_mat.height;
  float width = energies_mat.width;
  float** matrix = energies_mat.matrix;

  const int num_points = height * width;

  //Allocate energies
  float* energies;
  cudaMalloc(&energies, num_points * sizeof(float));
  cudaMemset(energies, 0, num_points * sizeof(float));

  const int row_size = width * sizeof(float);
  for(int y = 0; y < height; ++y) {
    int offset = y * width;
    cudaMemcpy(energies + offset, matrix[y], row_size, cudaMemcpyHostToDevice);
  }

  //Allocate energies
  float* min_energies;
  cudaMalloc(&min_energies, num_points * sizeof(float));

  //TODO new decomposition  
  int energy_blocks = width / THREADS_PER_BLOCK;
  for(int y = 0; y < height; ++y) {  
    min_cost<<<energy_blocks, THREADS_PER_BLOCK>>>(energies, width, y, min_energies);
  }

  float** out_energies = (float**) malloc(height * sizeof(float*));
  for(int y = 0; y < height; ++y) {
    out_energies[y] = (float*) malloc(width * sizeof(float));
    int offset = y * width;
    cudaMemcpy(out_energies[y], min_energies + offset, row_size, cudaMemcpyDeviceToHost);
  }

  cudaFree(energies);
  cudaFree(min_energies);

  //Should be no need for barriers thanks to final cuda memcpy
  return Matrix(out_energies, height, width);
}