#ifndef CUDA_LINK
#define CUDA_LINK

#include <iostream>
#include <ctime>
#include <vector>

#include <cuda_runtime.h>
#include "cuda_common/inc/helper_functions.h"
#include "cuda_common/inc/helper_cuda.h"

#include "cuda_kernels.h"

#include "globals.h"

class Quadrature;
class Mesh;
class XSection;
class SourceParams;

void reportGpuData();
int *alloc_gpuInt(int gpuId, int elements);
float *alloc_gpuFloat(int gpuId, int elements);
void release_gpu(int gpuId, float **gpus);
void updateCpuData(float *data_cpu, float *data_gpu1, int nx, int ny);

int launch_isoRayKernel(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const std::vector<RAY_T> *uflux, const SourceParams *params);

#endif // CUDA_LINK

