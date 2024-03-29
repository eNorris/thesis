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

#define CUDA_PI 3.14159
#define CUDA_4PI 12.5663706144
#define CUDA_4PI_INV 0.07957747154
//const SOL_T CUDA_PI = static_cast<SOL_T>(3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679);
//const SOL_T CUDA_4PI = static_cast<SOL_T>(4.0 * CUDA_PI);
//const SOL_T CUDA_4PI_INV = static_cast<SOL_T>(1.0 / CUDA_4PI);

void reportGpuData();
int *alloc_gpuInt(const int gpuId, const int elements, const int *data);
//float *alloc_gpuFloat(const int gpuId, const int elements, const float *data);
//void release_gpu(int gpuId, SOL_T *gpus);
//void release_gpu(int gpuId, int *gpus);

//void updateCpuData(int gpuId, SOL_T *data_cpu, SOL_T *data_gpu, size_t elements, int cpuOffset=0);
//void updateCpuData(int gpuId, int *data_cpu, int *data_gpu, size_t elements, int cpuOffset=0);

//template<typename T>
//void updateCpuDataBlocking(int gpuId, T *data_cpu, T *data_gpu, size_t elements, int cpuOffset=0);
//void updateCpuDataBlocking(int gpuId, int *data_cpu, int *data_gpu, size_t elements, int cpuOffset=0);

int launch_isoRayKernel(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar, std::vector<RAY_T> *uflux);
int launch_isoSolKernel(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar, const std::vector<RAY_T> *uflux, std::vector<SOL_T> *scalarFlux);

//template <class T>
//void reduce(int size, int threads, int blocks, T *d_idata, T *d_odata);

//void getNumBlocksAndThreads(int gpuId, int n, int maxBlocks, int maxThreads, int &blocks, int &threads);

template<typename T>
void updateCpuData(int gpuId, T *cpuData, T *gpuData, size_t elements, int cpuOffset=0)
{
    cudaError_t cudaerr;
    if((cudaerr = cudaSetDevice(gpuId)) != cudaSuccess)
        std::cout << "updateCpuData (T) failed to set the device with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;

    if((cudaerr = cudaMemcpyAsync(cpuData+cpuOffset, gpuData, elements*sizeof(T), cudaMemcpyDeviceToHost)) != cudaSuccess)
        std::cout << "updateCpuData (T) MemcpyAsync failed with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;
}

// HPP implementation
template<typename T>
void updateCpuDataBlocking(int gpuId, T *cpuData, T *gpuData, size_t elements, int cpuOffset=0)
{
    cudaError_t cudaerr;
    if((cudaerr = cudaSetDevice(gpuId)) != cudaSuccess)
        std::cout << "updateCpuDataBlocking (T) failed to set the device with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;

    if((cudaerr = cudaMemcpy(cpuData+cpuOffset, gpuData, elements*sizeof(T), cudaMemcpyDeviceToHost)) != cudaSuccess)
        std::cout << "updateCpuDataBlocking (T) Memcpy failed with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;
}


template<typename T>
T *alloc_gpuFloat(const int gpuId, const std::size_t elements, const T *cpuData)
{
    cudaError_t cudaerr;
    if((cudaerr = cudaSetDevice(gpuId)) != cudaSuccess)
        std::cout << "alloc_gpuFloat failed to set the device with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;

    T *gpuData;
    if((cudaerr = cudaMalloc(&gpuData, elements*sizeof(T))) != cudaSuccess)
        std::cout << "alloc_gpuFloat threw an error while allocating CUDA memory with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;

    if(cpuData != NULL)
    {
        if((cudaerr = cudaMemcpyAsync(gpuData, cpuData, elements*sizeof(T), cudaMemcpyHostToDevice)) != cudaSuccess)
            std::cout << "alloc_gpuFloat failed while copying data with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;
    }

    return gpuData;
}

template<typename T>
void release_gpu(int gpuId, T *gpuData)
{
    cudaError_t cudaerr;
    if((cudaerr = cudaSetDevice(gpuId)) != cudaSuccess)
        std::cout << "release_gpu (T) failed to set the device with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;

    if((cudaerr = cudaFree(gpuData)) != cudaSuccess)
        std::cout << "relase_gpu (T) threw an error while deallocating CUDA memory with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;
}


#endif // CUDA_LINK

