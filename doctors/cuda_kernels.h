#ifndef CUDA_KERNALS
#define CUDA_KERNALS

#include <cuda_runtime.h>
#include "globals.h"
#include <stdio.h>

#define CUDA_PI 3.14159
#define CUDA_4PI 12.5663706144
#define CUDA_4PI_INV 0.07957747154

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
/*
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};
*/

__global__ void isoRayKernel(
        RAY_T *uflux,
        float *xNodes, float *yNodes, float *zNodes,
        float *dx, float *dy, float *dz,
        int *zoneId,
        float *atomDensity,
        float *tot1d,
        float *srcStrength,
        int groups,
        int Nx, int Ny, int Nz,
        float sx, float sy, float sz,
        int srcIndxX, int srcIndxY, int srcIndxZ
        );

__global__ void isoSolKernel(
        SOL_T *scalarFlux, SOL_T *tempFlux,
        SOL_T *totalSource,
        float *totXs1d, float *scatxs2d,
        float *Axy, float *Axz, float *Ayz,
        int *zoneId, float *atomDensity, float *vol,
        float *mu, float *eta, float *xi, float *wt,
        SOL_T *outboundFluxX, SOL_T *outboundFluxY, SOL_T *outboundFluxZ,
        int ie, int iang,
        int Nx, int Ny, int Nz, int groups, int angleCount, int pn,
        int dix, int diy, int diz,
        int startIndx, int voxThisLevel, int *gpuIdxToMesh);

__global__ void isoSrcKernel(
        RAY_T *uFlux,
        SOL_T *extSource,
        float *vol, float *atomDensity, int *zoneId,
        float *scatxs2d,
        int voxels, int groups, int pn, int highestEnergyGroup, int sinkGroup,
        int Nx, int Ny, int Nz);

__global__ void zeroKernel(int elements, SOL_T *ptr);
__global__ void zeroKernelMesh(int Nx, int Ny, int Nz, SOL_T *ptr);
__global__ void zeroKernelMeshEnergy(int groups, int Nx, int Ny, int Nz, SOL_T *ptr);

__global__ void downscatterKernel(
        SOL_T *totalSource,
        int highestEnergyGroup, int sinkGroup,
        int Nx, int Ny, int Nz, int groups, int pn,
        int *zoneId,
        SOL_T *scalarFlux,
        float *scatxs2d,
        float *atomDensity, float *vol,
        SOL_T *extSource);

__global__ void clearSweepKernel(
        SOL_T *cFlux, SOL_T *tempFlux,
        int Nx, int Ny, int Nz, int ie);

//__global__ void isoDiffKernel();

//template <class T, unsigned int blockSize, bool nIsPow2>
//__global__ void
//reduce6(T *g_idata, T *g_odata, unsigned int n);

#endif // CUDA_KERNALS

