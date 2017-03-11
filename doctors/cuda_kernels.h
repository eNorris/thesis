#ifndef CUDA_KERNALS
#define CUDA_KERNALS

#include <cuda_runtime.h>

__global__ void isoRayKernel(float *uflux,
                             int xIndxStart, int yIndxStart, int zIndxStart,
                             float *xNodes, float *yNodes, float zNodes,
                             float *dx, float *dy, float *dz,
                             int *zoneId,
                             float *atomDensity,
                             int groups,
                             float *tot1d,
                             flost sx, float sy, float sz,
                             srcIndxX, int srcIndxY, int srcIndxZ,
                             float *srcStrength);

__global__ void isoSolKernel(float *data1, float *data2, int nx, int ny);

#endif // CUDA_KERNALS

