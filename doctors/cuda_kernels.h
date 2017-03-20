#ifndef CUDA_KERNALS
#define CUDA_KERNALS

#include <cuda_runtime.h>
#include "globals.h"
#include <stdio.h>

#define CUDA_PI 3.14159
#define CUDA_4PI 12.5663706144
#define CUDA_4PI_INV 0.07957747154

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
        int Nx, int Ny, int Nz, int groups, int angleCount, int pn
        );

__global__ void isoSrcKernel(
        float *uFlux,
        float *extSource,
        float *vol, float *atomDensity, int *zoneId,
        float *scatxs2d,
        int voxels, int groups, int pn, int highestEnergyGroup, int sinkGroup,
        int Nx, int Ny, int Nz);

__global__ void zeroKernel(int Nx, int Ny, int Nz, float *ptr);

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
        float *preFlux, float *tempFlux,
        int Nx, int Ny, int Nz);

#endif // CUDA_KERNALS

