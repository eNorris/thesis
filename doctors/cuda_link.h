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
class SolverParams;

void reportGpuData();
int *alloc_gpuInt(const int gpuId, const int elements, const int *data);
float *alloc_gpuFloat(const int gpuId, const int elements, const float *data);
void release_gpu(int gpuId, float *gpus);
void release_gpu(int gpuId, int *gpus);
void updateCpuData(int gpuId, float *data_cpu, float *data_gpu, size_t elements, int cpuOffset=0);
void updateCpuData(int gpuId, int *data_cpu, int *data_gpu, size_t elements, int cpuOffset=0);

int launch_isoRayKernel(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar, std::vector<RAY_T> *uflux);

#endif // CUDA_LINK

