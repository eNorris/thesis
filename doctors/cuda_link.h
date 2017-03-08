#ifndef CUDA_LINK
#define CUDA_LINK

#include <iostream>
#include <ctime>

#include <cuda_runtime.h>
#include "cuda_common/inc/helper_functions.h"
#include "cuda_common/inc/helper_cuda.h"

#include "cuda_kernels.h"

using std::cout;
using std::endl;

float **init_gpu(int Nx, int Ny, float *cpu_data);
void updateCpuData(float *data_cpu, float *data_gpu1, int nx, int ny);

int launch_diffusionKernel(int nx, int ny, float *gpu1, float *gpu2);

#endif // CUDA_LINK

