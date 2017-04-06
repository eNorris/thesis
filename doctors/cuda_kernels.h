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
        float *uflux,
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
        float *scalarFlux, float *tempFlux,
        float *totalSource,
        float *totXs1d, float *scatxs2d,
        float *Axy, float *Axz, float *Ayz,
        int *zoneId, float *atomDensity, float *vol,
        float *mu, float *eta, float *xi, float *wt,
        float *outboundFluxX, float *outboundFluxY, float *outboundFluxZ,
        int ie, int iang,
        int Nx, int Ny, int Nz, int groups, int angleCount, int pn,
        int dix, int diy, int diz,
        int startIndx, int voxThisLevel, int *gpuIdxToMesh);

__global__ void isoSrcKernel(
        float *uFlux,
        float *extSource,
        float *vol, float *atomDensity, int *zoneId,
        float *scatxs2d,
        int voxels, int groups, int pn, int highestEnergyGroup, int sinkGroup,
        int Nx, int Ny, int Nz);

__global__ void zeroKernel(int elements, float *ptr);
__global__ void zeroKernelMesh(int Nx, int Ny, int Nz, float *ptr);
__global__ void zeroKernelMeshEnergy(int groups, int Nx, int Ny, int Nz, float *ptr);

__global__ void downscatterKernel(
        float *totalSource,
        int highestEnergyGroup, int sinkGroup,
        int Nx, int Ny, int Nz, int groups, int pn,
        int *zoneId,
        float *scalarFlux,
        float *scatxs2d,
        float *atomDensity, float *vol,
        float *extSource);

__global__ void clearSweepKernel(
        float *cFlux, float *tempFlux,
        int Nx, int Ny, int Nz, int ie);

//__global__ void isoDiffKernel();

//template <class T, unsigned int blockSize, bool nIsPow2>
//__global__ void
//reduce6(T *g_idata, T *g_odata, unsigned int n);

#endif // CUDA_KERNALS

