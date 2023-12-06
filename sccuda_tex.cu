#include "scexec.hpp"
#include "matrix.hpp"
#include "image.hpp"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

constexpr NUM_COLORS = 3;
constexpr int BLOCK_DIM = 16;
constexpr int THREADS_PER_BLOCK = BLOCK_DIM * BLOCK_DIM;

#define TRIANGLE_BASE 60
#define TRIANGLE_SIZE 930
#define TRIANGLE_HEIGHT 30

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

cudaTextureObject_t imageToTex(Image& image) {
  int width = image.width;
  int height = image.height;
  float** image_data = image.data;

  int num_points = height * width;

  cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray_t imageArray;
  cudaMallocArray(&imageArray, &channelDesc, height, width);

  float* image_flat = malloc(num_points * NUM_COLORS * sizeof(float));
  for(int y = 0; y < height; ++y) {
    memcpy(image_flat + y * width * NUM_COLORS, image_data[y], width * NUM_COLORS * sizeof(float));
  }

  int spitch = sizeof(float) * width * NUM_COLORS
  int byte_width = sizeof(float) * width * NUM_COLORS

  cudaMemcpy2DToArray(imageArray, 0, 0, image_flat, spitch, width, height, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  free(image_flat);

  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.type = cudaResourceTypeArray;
  resDesc.res.array = imageArray;
  
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
  cudaDeviceSynchronize();

  //TODO how to free CUDA array?
  return texObj;
}

cudaTextureObject_t energiesToTex(Matrix& energies) {
  int width = energies.width;
  int height = energies.height;
  float** energy_data = energies.matrix;

  int num_points = height * width;

  cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray_t energyArray;
  cudaMallocArray(&energyArray, &channelDesc, height, width);

  float* energy_flat = malloc(num_points * sizeof(float));
  for(int y = 0; y < height; ++y) {
    memcpy(energy_flat + y * width, energy_data[y], width * sizeof(float));
  }

  int spitch = sizeof(float) * width;
  int byte_width = sizeof(float) * width;

  cudaMemcpy2DToArray(energyArray, 0, 0, energy_flat, spitch, width, height, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  free(energy_flat);

  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.type = cudaResourceTypeArray;
  resDesc.res.array = energyArray;
  
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
  cudaDeviceSynchronize();

  //TODO how to free CUDA array?
  return texObj;
}

__device__ float x_gradient(cudaTextureObject_t tex, float x, float y) {
  float3 left = tex2d<float3>(tex, x-1, y);
  float3 right = tex2d<float3>(tex, x+1, y);

  float r = powf(abs(left.x - right.x), 2);
  float g = powf(abs(left.y - right.y), 2);
  float b = powf(abs(left.z - right.z), 2);

  return r + g + b;
}

__device__ float y_gradient(cudaTextureObject_t tex, float x, float y) {
  float3 below = tex2d<float3>(tex, x, y-1);
  float3 above = tex2d<float3>(tex, x, y+1);

  float r = powf(abs(below.x - above.x), 2);
  float g = powf(abs(below.y - above.y), 2);
  float b = powf(abs(below.z - above.z), 2);

  return r + g + b;
}

__global__ void dual_gradient(cudaTextureObject_t tex, int height, int width, float* energies) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < width && y < height) {
    int energy_idx = y * width + x;
    float x_grad = x_gradient(tex, x, y);
    float y_grad = y_gradient(tex, x, y);

    energies[energy_idx] = x_grad + y_grad;
  }
}

/**
* Uses dual gradient energy to create an energy matrix of the image pixels
* @param image 3d matrix representing the image (height, width, colors)
* @param height the image height in pixels
* @param width the image width in pixels
* @return a 2d float matrix (height, width) representing the energy of each pixel
*/
__host__ Matrix compute_energy_mat_cuda_tex(Image& image) {
  find_cuda_device();

  int width = image.width;
  int height = image.height;

  int num_points = height * width;
  //Send image to GPU
  //Allocate image
  cudaTextureObject_t tex = imageToTex(image);
  
  float* energies;
  cudaMalloc(&energies, num_points * sizeof(float));
  cudaMemset(energies, 0, num_points * sizeof(float));

  //Block decomposition
  dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
  dim3 num_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
                  (height + threads_per_block.y - 1) / threads_per_block.y);

  //launch kernel
  dual_gradient<<<num_blocks, threads_per_block>>>(tex, height, width, energies);

  //Free variables
  cudaResourceDesc res_desc;
  cudaGetTextureObjectResourceDesc(&res_desc, tex);

  cudaArray_t tex_array = res_desc.array;

  cudaDestroyTextureObject(tex);
  cudaFreeArray(tex_array);

  float** out_energies = (float**)malloc(height * sizeof(float*));
  for(int y = 0; y < height; ++y) {
    out_energies[y] = (float*)malloc(width * sizeof(float));
    int offset = y * width * sizeof(float);
    cudaMemcpy(out_energies[y], energies + offset, width * sizeof(float), cudaMemcpyDeviceToHost);
  }

  cudaFree(energies);

  return Matrix(out_energies, height, width);
}

__device__ get_energy(cudaTextureObject_t energies,  int x, int y) {
  return tex2D<float>(energies, x, y);
}

__global__ void min_cost(cudaTextureObject_t energies, int height, int width, int row, float* min_energies) {
  //TODO triangle decomposition
  int y = row;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  float min_energy_above = 0;
  if(y > 0) {
    min_energy_above = get_energy(energies, x, y - 1);
    if(x > 0) {
      float left = get_energy(energies, x - 1, y - 1);
      min_energy_above = min(min_energy_above, left);
    }
    if(x < width - 1) {
      float right = get_energy(energies, x + 1, y - 1);
      min_energy_above = min(min_energy_above, right);
    }
  }

  return min_energy_above + get_energy(energies, height, width, x, y);
}

__host__ Matrix compute_min_cost_mat_cuda_tex(Matrix& energies) {
  int height = energies.height;
  int width = energies.width;

  find_cuda_device();
  const int num_points = height * width;

  //Send image to GPU
  //Allocate image
  cudaTextureObject_t tex = energyToTex(energies);
  
  float* min_energies;
  cudaMalloc(&min_energies, num_points * sizeof(float));

  //Block decomposition
  int num_blocks = width / THREADS_PER_BLOCK;

  //launch kernel
  for(int y = 0; y < height; ++y) {
    min_cost<<<num_blocks, THREADS_PER_BLOCK>>>(tex, height, width, row, min_energies);
  }

  //Free variables
  cudaResourceDesc res_desc;
  cudaGetTextureObjectResourceDesc(&res_desc, tex);

  cudaArray_t tex_array = res_desc.array;

  cudaDestroyTextureObject(tex);
  cudaFreeArray(tex_array);

  float** out_energies = (float**)malloc(height * sizeof(float*));
  for(int y = 0; y < height; ++y) {
    out_energies[y] = (float*)malloc(width * sizeof(float));
    int offset = y * width * sizeof(float);
    cudaMemcpy(out_energies[y], min_energies + offset, width * sizeof(float), cudaMemcpyDeviceToHost);
  }

  cudaFree(min_energies);

  return Matrix(out_energies, height, width);
}