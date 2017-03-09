#ifndef CUDA_KERNALS
#define CUDA_KERNALS

#include <cuda_runtime.h>

__global__ void isoRayKernel(float *data1, float *data2, int nx, int ny);

__global__ void isoSolKernel(float *data1, float *data2, int nx, int ny);

#endif // CUDA_KERNALS

