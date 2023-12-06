#include "scexec.hpp"
#include "image.hpp"
#include "matrix.hpp"
#include "timing.hpp"

#include <iostream>
#include <assert.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define BLOCK_DIM 16
#define THREADS_PER_BLOCK 256
#define NUM_COLORS 3

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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
  int x_up = x == 0 ? height - 1 : x - 1;
  int x_down = x == height - 1 ? 0 : x + 1;

  float* left = get_pixel(image, width, x_up, y);
  float* right = get_pixel(image, width, x_down, y);

  float r = powf(left[0] - right[0], 2);
  float g = powf(left[1] - right[1], 2);
  float b = powf(left[2] - right[2], 2);

  return sqrtf(r + g + b);
}

__device__ float y_gradient(float* image, int height, int width, int x, int y) {
  int y_left = y == 0 ? width - 1 : y - 1;
  int y_right = y == width - 1 ? 0 : y + 1;

  float* above = get_pixel(image, width, x, y_left);
  float* below = get_pixel(image, width, x, y_right);

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
    energies[energy_idx] = x_grad + y_grad + 1.0;
  }
}

__device__ float get_energy(float* energies, int width, int x, int y) {
  int idx = y * width + x;
  return energies[idx];
}

__global__ void min_cost(float* energies, int width, int row, float* min_energies) {
  int y = row;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if(x < width) {
    float min_energy_above = 0;
    if(y > 0) {
      for(int k = x - 1; k <= x + 1; k++) {
        if (k >= 0 && k < width) {
          float energy = get_energy(min_energies, width, k, y - 1);
          if (min_energy_above == 0 || energy < min_energy_above) {
            min_energy_above = energy;
          }
        }
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
__host__ void compute_min_cost_mat_direct_cuda(timing_t& timing, Image& image, int new_width) {
  find_cuda_device();

  int height = image.height;
  int width = image.width;

  // timer events
  cudaEvent_t start_e, stop_e;
  cudaEventCreate(&start_e);
  cudaEventCreate(&stop_e);
  cudaEvent_t start_m, stop_m;
  cudaEventCreate(&start_m);
  cudaEventCreate(&stop_m);

  //Send image to GPU
  const int num_points = height * width;

  //Allocate image
  float* image_flat;
  gpuErrchk(cudaMalloc(&image_flat, num_points * NUM_COLORS * sizeof(float)));

  //Allocate energies
  float* energies;
  gpuErrchk(cudaMalloc(&energies, num_points * sizeof(float)));

  float* min_energies;
  gpuErrchk(cudaMalloc(&min_energies, num_points * sizeof(float)));

  int temp_width = width;
  int pixel_row_size = temp_width * NUM_COLORS * sizeof(float);
  while(temp_width > new_width) {
    float** image_data = image.image;

    for(int r = 0; r < height; ++r) {
      int img_idx = r * temp_width * NUM_COLORS;
      cudaMemcpy(image_flat + img_idx, image_data[r], pixel_row_size, cudaMemcpyHostToDevice);
    }
    cudaMemset(energies, 0, num_points * sizeof(float));

    //Block decomposition
    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    dim3 num_blocks((temp_width + threads_per_block.x - 1) / threads_per_block.x,
                    (temp_width + threads_per_block.y - 1) / threads_per_block.y);

    //launch kernel
    cudaEventRecord(start_e);
    dual_gradient<<<num_blocks, threads_per_block>>>(image_flat, height, temp_width, energies);
    cudaEventRecord(stop_e);
    cudaEventSynchronize(stop_e);
    float e_milliseconds = 0;
    cudaEventElapsedTime(&e_milliseconds, start_e, stop_e);
    timing.energy_time += e_milliseconds / 1000;

    //TODO new decomposition  
    cudaEventRecord(start_m);
    int energy_blocks = (temp_width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    for(int y = 0; y < height; ++y) {  
      min_cost<<<energy_blocks, THREADS_PER_BLOCK>>>(energies, temp_width, y, min_energies);
    }
    cudaEventRecord(stop_m);
    cudaEventSynchronize(stop_m);
    float m_milliseconds = 0;
    cudaEventElapsedTime(&m_milliseconds, start_m, stop_m);
    timing.min_cost_time += m_milliseconds / 1000;

    float** out_energies = (float**) malloc(height * sizeof(float*));
    int row_size = temp_width * sizeof(float);
    for(int y = 0; y < height; y++) {
      out_energies[y] = (float*) malloc(row_size);
      int offset = y * temp_width;
      gpuErrchk(cudaMemcpy(out_energies[y], min_energies + offset, row_size, cudaMemcpyDeviceToHost));
    }

    Matrix min_cost_mat = Matrix(out_energies, height, temp_width);
    int *seam = find_seam(min_cost_mat);
    image.remove_seam(seam);

    delete seam;
    // delete &min_cost_mat;

    temp_width--;
  }

  gpuErrchk(cudaFree(image_flat));
  gpuErrchk(cudaFree(energies));
  gpuErrchk(cudaFree(min_energies));
}

//
// non-direct stuff
//

__host__ Matrix compute_energy_mat_cuda(timing_t& timing, Image& image) {
  find_cuda_device();

  int height = image.height;
  int width = image.width;
  float** image_data = image.image;

  cudaEvent_t start_e, stop_e;
  cudaEventCreate(&start_e);
  cudaEventCreate(&stop_e);

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
  dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
  dim3 num_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
                  (height + threads_per_block.y - 1) / threads_per_block.y);

  //launch kernel
  cudaEventRecord(start_e);
  dual_gradient<<<num_blocks, threads_per_block>>>(image_flat, height, width, energies);
  cudaEventRecord(stop_e);
  cudaEventSynchronize(stop_e);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start_e, stop_e);
  timing.energy_time += milliseconds / 1000;

  float** out_energies = (float**) malloc(height * sizeof(float*));
  const int row_size = width * sizeof(float);
  for(int y = 0; y < height; y++) {
    out_energies[y] = (float*) malloc(width * sizeof(float));
    int offset = y * width;
    cudaMemcpy(out_energies[y], energies + offset, row_size, cudaMemcpyDeviceToHost);
  }

  cudaFree(energies);
  gpuErrchk(cudaFree(image_flat));

  //Should be no need for barriers thanks to final cuda memcpy
  return Matrix(out_energies, height, width);
}

__host__ Matrix compute_min_cost_mat_cuda(timing_t& timing, Matrix& energies_mat) {
  find_cuda_device();

  int height = energies_mat.height;
  int width = energies_mat.width;
  float** matrix = energies_mat.matrix;

  cudaEvent_t start_m, stop_m;
  cudaEventCreate(&start_m);
  cudaEventCreate(&stop_m);

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
  cudaEventRecord(start_m);
  int energy_blocks = (width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  for(int y = 0; y < height; ++y) {  
    min_cost<<<energy_blocks, THREADS_PER_BLOCK>>>(energies, width, y, min_energies);
  }
  cudaEventRecord(stop_m);
  cudaEventSynchronize(stop_m);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start_m, stop_m);
  timing.min_cost_time += milliseconds / 1000;

  float** out_energies = (float**) malloc(height * sizeof(float*));
  for(int y = 0; y < height; y++) {
    out_energies[y] = (float*) malloc(width * sizeof(float));
    int offset = y * width;
    cudaMemcpy(out_energies[y], min_energies + offset, row_size, cudaMemcpyDeviceToHost);
  }

  cudaFree(energies);
  cudaFree(min_energies);

  //Should be no need for barriers thanks to final cuda memcpy
  return Matrix(out_energies, height, width);
}
