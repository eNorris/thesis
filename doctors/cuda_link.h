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

float **init_gpu(int Nx, int Ny, float *cpu_data);
void updateCpuData(float *data_cpu, float *data_gpu1, int nx, int ny);

int launch_isoRayKernel(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const std::vector<RAY_T> *uflux, const SourceParams *params);

#endif // CUDA_LINK

