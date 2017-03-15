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

__global__ void isoSolKernel(float *data1, float *data2, int nx, int ny);

#endif // CUDA_KERNALS

