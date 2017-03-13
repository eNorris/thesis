#ifndef CUDA_KERNALS
#define CUDA_KERNALS

#include <cuda_runtime.h>
#include "globals.h"
#include <stdio.h>

const SOL_T CUDA_PI = static_cast<SOL_T>(3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679);
const SOL_T CUDA_4PI = static_cast<SOL_T>(4.0 * CUDA_PI);
const SOL_T CUDA_4PI_INV = static_cast<SOL_T>(1.0 / CUDA_4PI);

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

__global__ void isoSolKernel(float *data1, float *data2, int nx, int ny);

#endif // CUDA_KERNALS

