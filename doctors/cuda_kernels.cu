#include "cuda_kernels.h"

__global__ void testKernel4(float *prevdata, float *data, int nx, int ny)
{
    float t = 0.0f;
    float c = 0.0f;

    int indx = ny*blockIdx.x + threadIdx.x;

    if(blockIdx.x > 0)
    {
        t += (prevdata[indx - ny] - prevdata[indx]);
        c += 1.0f;
    }
    if(blockIdx.x < nx-1)
    {
        t += (prevdata[indx + ny] - prevdata[indx]);
        c+=1.0f;
    }
    if(threadIdx.x > 0)
    {
        t += (prevdata[indx-1] - prevdata[indx]);
        c+=1.0f;
    }
    if(threadIdx.x < nx-1)
    {
        t += (prevdata[indx+1] - prevdata[indx]);
        c+=1.0f;
    }

    data[indx] = prevdata[indx] + t/c*0.5;

    if(threadIdx.x == 0)
        data[indx] = 1.0;

    prevdata[indx] = data[indx];

    return;
}
