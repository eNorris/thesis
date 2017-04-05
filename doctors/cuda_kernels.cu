#include "cuda_kernels.h"



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
        )
{
    //int xIndxStart = blockIdx.x;
    //int yIndxStart = 2*blockIdx.y + threadIdx.y;
    //int zIndxStart = threadIdx.x;

    int xIndxStart = blockIdx.x;
    int yIndxStart = blockIdx.y;
    int zIndxStart = threadIdx.x;

    const unsigned short DIRECTION_X = 1;
    const unsigned short DIRECTION_Y = 2;
    const unsigned short DIRECTION_Z = 3;

    RAY_T tiny = 1.0E-35f;
    RAY_T huge = 1.0E35f;

    int ir0 = xIndxStart*Ny*Nz + yIndxStart*Nz + zIndxStart;

    float *meanFreePaths = new float[groups];

    // This runs for a single voxel
    RAY_T x = xNodes[xIndxStart] + dx[xIndxStart]/2;
    RAY_T y = yNodes[yIndxStart] + dy[yIndxStart]/2;
    RAY_T z = zNodes[zIndxStart] + dz[zIndxStart]/2;

    if(xIndxStart == srcIndxX && yIndxStart == srcIndxY && zIndxStart == srcIndxZ)  // End condition
    {

        RAY_T srcToCellDist = sqrt((x-sx)*(x-sx) + (y-sy)*(y-sy) + (z-sz)*(z-sz));
        unsigned int zid = zoneId[ir0];
        RAY_T xsval;
        for(unsigned int ie = 0; ie < groups; ie++)
        {
            xsval = tot1d[zid*groups + ie] * atomDensity[ir0];
            uflux[ie*Nx*Ny*Nz + ir0] = srcStrength[ie] * exp(-xsval*srcToCellDist) / (CUDA_4PI * srcToCellDist * srcToCellDist);
        }
        return;
    }

    // Start raytracing through the geometry
    unsigned int xIndx = xIndxStart;
    unsigned int yIndx = yIndxStart;
    unsigned int zIndx = zIndxStart;

    RAY_T srcToCellX = sx - x;
    RAY_T srcToCellY = sy - y;
    RAY_T srcToCellZ = sz - z;

    RAY_T srcToCellDist = sqrt(srcToCellX*srcToCellX + srcToCellY*srcToCellY + srcToCellZ*srcToCellZ);
    RAY_T srcToPtDist;  // Used later in the while loop

    RAY_T xcos = srcToCellX/srcToCellDist;  // Fraction of direction biased in x-direction, unitless
    RAY_T ycos = srcToCellY/srcToCellDist;
    RAY_T zcos = srcToCellZ/srcToCellDist;

    int xBoundIndx = (xcos >= 0 ? xIndx+1 : xIndx);
    int yBoundIndx = (ycos >= 0 ? yIndx+1 : yIndx);
    int zBoundIndx = (zcos >= 0 ? zIndx+1 : zIndx);

    // Clear the MPF array to zeros
    for(unsigned int i = 0; i < groups; i++)
        meanFreePaths[i] = 0.0f;

    bool exhaustedRay = false;
    int iter = 0;
    while(!exhaustedRay)
    {
        int ir = xIndx*Ny*Nz + yIndx*Nz + zIndx;

        srcToCellX = sx - x;
        srcToCellY = sy - y;
        srcToCellZ = sz - z;
        srcToPtDist = sqrt(srcToCellX*srcToCellX + srcToCellY*srcToCellY + srcToCellZ*srcToCellZ);
        xcos = srcToCellX/srcToPtDist;  // Fraction of direction biased in x-direction, unitless
        ycos = srcToCellY/srcToPtDist;
        zcos = srcToCellZ/srcToPtDist;

        // Determine the distance to cell boundaries
        RAY_T tx = (fabs(xcos) < tiny ? huge : (xNodes[xBoundIndx] - x)/xcos);  // Distance traveled [cm] when next cell is
        RAY_T ty = (fabs(ycos) < tiny ? huge : (yNodes[yBoundIndx] - y)/ycos);  //   entered traveling in x direction
        RAY_T tz = (fabs(zcos) < tiny ? huge : (zNodes[zBoundIndx] - z)/zcos);

        // Determine the shortest distance traveled [cm] before _any_ surface is crossed
        RAY_T tmin;
        unsigned short dirHitFirst;

        if(tx < ty && tx < tz)
        {
            tmin = tx;
            dirHitFirst = DIRECTION_X;
        }
        else if(ty < tz)
        {
            tmin = ty;
            dirHitFirst = DIRECTION_Y;
        }
        else
        {
            tmin = tz;
            dirHitFirst = DIRECTION_Z;
        }

        // Update mpf array
        unsigned int zid = zoneId[ir];
        for(unsigned int ie = 0; ie < groups; ie++)
        {
            //                   [cm] * [b] * [atom/b-cm]
            meanFreePaths[ie] += tmin * tot1d[zid*groups + ie] * atomDensity[ir];
        }

        // Update cell indices and positions
        if(dirHitFirst == DIRECTION_X) // x direction
        {
            x = xNodes[xBoundIndx];
            y += tmin*ycos;
            z += tmin*zcos;
            if(xcos >= 0)
            {
                xIndx++;
                xBoundIndx++;
            }
            else
            {
                xIndx--;
                xBoundIndx--;
            }
        }
        else if(dirHitFirst == DIRECTION_Y) // y direction
        {
            x += tmin*xcos;
            y = yNodes[yBoundIndx];
            z += tmin*zcos;
            if(ycos >= 0)
            {
                yIndx++;
                yBoundIndx++;
            }
            else
            {
                yIndx--;
                yBoundIndx--;
            }
        }
        else if(dirHitFirst == DIRECTION_Z) // z direction
        {
            x += tmin*xcos;
            y += tmin*ycos;
            z = zNodes[zBoundIndx];
            if(zcos >= 0)
            {
                zIndx++;
                zBoundIndx++;
            }
            else
            {
                zIndx--;
                zBoundIndx--;
            }
        }

        if((xIndx == srcIndxX && yIndx == srcIndxY && zIndx == srcIndxZ) || exhaustedRay)
        {
            RAY_T finalDist = sqrt((x-sx)*(x-sx) + (y-sy)*(y-sy) + (z-sz)*(z-sz));

            for(unsigned int ie = 0; ie < groups; ie++)
            {
                //       [#]       = [cm] * [b] * [1/cm-b]
                meanFreePaths[ie] += finalDist * tot1d[zid*groups + ie] * atomDensity[ir];
            }

            exhaustedRay = true;
        }
        iter++;

    } // End of while loop

    for(unsigned int ie = 0; ie < groups; ie++)
    {
        RAY_T flx = srcStrength[ie] * exp(-meanFreePaths[ie]) / (CUDA_4PI * srcToCellDist * srcToCellDist);
        uflux[ie*Nx*Ny*Nz + ir0] = static_cast<SOL_T>(flx);  //srcStrength * exp(-meanFreePaths[ie]) / (4 * M_PI * srcToCellDist * srcToCellDist);
    }

    delete [] meanFreePaths;
}

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
        int startIndx, int voxThisSubSweep, int *gpuIdxToMesh)
{

    int iGpu = blockIdx.x*Ny*Nz + blockIdx.y*Nz + threadIdx.x;
    if(iGpu >= voxThisSubSweep)
        return;

    int ir = gpuIdxToMesh[startIndx + iGpu];

    int ix = ir / (Ny*Nz);
    int iy = (ir-ix*Ny*Nz) / Nz;
    int iz = ir - ix*Ny*Nz - iy*Nz;

    printf("iGpu=%d, voxThisSubSweep=%d, startIndx=%d, ir=%d, ix=%d, iy=%d, iz=%d, iS=%d\n", iGpu, voxThisSubSweep, startIndx, ir, ix, iy, iz, ix+iy+iz);

    if(ir == 33288)
    {
        printf("ix=%d, iy=%d, iz=%d", ix, iy, iz);
    }

    // Reverse directions as necessary
    if(dix == -1)
        ix = Nx-1 - ix;

    if(diy == -1)
        iy = Ny-1 - iy;

    if(diz == -1)
        iz = Nz-1 - iz;

    // Recompute the voxel index after reversing directions
    ir = ix*Ny*Nz + iy*Nz + iz;

    float influxX, influxY, influxZ;

    int zid = zoneId[ir];  // Get the zone id of this element

    // Handle the x influx
    if(dix == 1)                                       // Approach x = 0 -> xMesh
    {
        if(ix == 0)                                               // If this is a boundary cell
            influxX = 0.0f;                                       // then the in-flux is zero
        else                                                      // otherwise
            influxX = outboundFluxX[ir - Ny*Nz];  // the in-flux is the out-flux from the previous cell
    }
    else                                                          // Approach x = xMesh-1 -> 0
    {
        if(ix == Nx - 1)
            influxX = 0.0f;
        else
            influxX = outboundFluxX[ir + Ny*Nz];
    }

    // Handle the y influx
    if(diy == 1)                                       // Approach y = 0 -> yMesh
    {
        if(iy == 0)
            influxY = 0.0f;
        else
            influxY = outboundFluxY[ir - Nz];
    }
    else                                                          // Approach y = yMesh-1 -> 0
    {
        if(iy == Ny - 1)
            influxY = 0.0f;
        else
            influxY = outboundFluxY[ir + Nz];
    }

    // Handle the z influx
    if(diz == 1)
    {
        if(iz == 0)
            influxZ = 0.0f;
        else
            influxZ = outboundFluxZ[ir - 1];
    }
    else
    {
        if(iz == Nz - 1)
            influxZ = 0.0f;
        else
            influxZ = outboundFluxZ[ir + 1];
    }

    SOL_T inscatter = CUDA_4PI_INV*scalarFlux[ie*Nx*Ny*Nz + ir] * scatxs2d[zid*groups*groups*pn + ie*groups * pn +  ie*groups] * atomDensity[ir] * vol[ir];

    SOL_T numer = totalSource[ir] +  inscatter +                                                                                            // [#]
            Ayz[ie*angleCount*Ny*Nz + iang*Ny*Nz + iy*Nz + iz] * influxX +  // [cm^2 * #/cm^2]  The 2x is already factored in
            Axz[ie*angleCount*Nx*Nz + iang*Nx*Nz + ix*Nz + iz] * influxY +
            Axy[ie*angleCount*Nx*Ny + iang*Nx*Ny + ix*Ny + iy] * influxZ;
    SOL_T denom = vol[ir] * totXs1d[zid*groups + ie] * atomDensity[ir] +                               // [cm^3] * [b] * [1/b-cm]
            Ayz[ie*angleCount*Ny*Nz + iang*Ny*Nz + iy*Nz + iz] +            // [cm^2]
            Axz[ie*angleCount*Nx*Nz + iang*Nx*Nz + ix*Nz + iz] +
            Axy[ie*angleCount*Nx*Ny + iang*Nx*Ny + ix*Ny + iy];

    //   [#/cm^2] = [#]  / [cm^2]
    SOL_T angFlux = numer/denom;

    if(ix == 32 && iy == 32 && iz == 8)
    {
        printf("psi=%f, totSrc=%f, inScat=%f, numer=%f, denom=%f", angFlux, totalSource[ir], inscatter, numer, denom);
    }

    //angularFlux[ie*ejmp + iang*ajmp + ix*xjmp + iy*yjmp + iz] = angFlux;

    outboundFluxX[ir] = 2*angFlux - influxX;
    outboundFluxY[ir] = 2*angFlux - influxY;
    outboundFluxZ[ir] = 2*angFlux - influxZ;

    // Sum all the angular fluxes
    tempFlux[ir] += wt[iang]*angFlux;
}

__global__ void isoSrcKernel(
        float *uFlux,
        float *extSource,
        float *vol, float *atomDensity, int *zoneId,
        float *scatxs2d,
        int voxels, int groups, int pn, int highestEnergyGroup, int sinkGroup,
        int Nx, int Ny, int Nz)
{
    int ir = blockIdx.x*Ny*Nz + blockIdx.y*Nz + threadIdx.x;
    //for(unsigned int iep = highestEnergyGroup; iep <= sinkGroup; iep++)  // Sink energy
    //{
    //if(sinkGroup == 17)
    //    printf("Called %d -> %d @ %d (%d, %d, %d): uflux=%f, v=%f, s=%f, p=%f\n",
    //           iep, sinkGroup, ir, blockIdx.x, blockIdx.y, threadIdx.x,
    //           uFlux[iep*voxels + ir], vol[ir], scatxs2d[zoneId[ir]*groups*groups*pn + iep*groups*pn + sinkGroup*pn], atomDensity[ir]);
    //extSource[sinkGroup*voxels + ir] += uFlux[iep*voxels + ir] * vol[ir] * scatxs2d[zoneId[ir]*groups*groups*pn + iep*groups*pn + sinkGroup*pn] * atomDensity[ir];
    extSource[sinkGroup*voxels + ir] = uFlux[sinkGroup*voxels + ir] * vol[ir] * scatxs2d[zoneId[ir]*groups*groups*pn + sinkGroup*groups*pn + sinkGroup*pn] * atomDensity[ir];
    //}
}
__global__ void zeroKernelMesh(int Nx, int Ny, int Nz, float *ptr)
{
    int ir = blockIdx.x*Ny*Nz + blockIdx.y*Nz + threadIdx.x;
    ptr[ir] = 0.0f;
}

__global__ void zeroKernelMeshEnergy(int groups, int Nx, int Ny, int Nz, float *ptr)
{
    int ir = blockIdx.x*Ny*Nz + blockIdx.y*Nz + threadIdx.x;
    for(unsigned int ie = 0; ie < groups; ie++)
    {
        //printf("Indx: %d = %d, %d, %d, %d\n", groups*Nx*Ny*Nz + ir, i);
        ptr[ie*Nx*Ny*Nz + ir] = 0.0f;
    }
}

__global__ void downscatterKernel(
        float *totalSource,
        int highestEnergyGroup, int sinkGroup,
        int Nx, int Ny, int Nz, int groups, int pn,
        int *zoneId,
        float *scalarFlux,
        float *scatxs2d,
        float *atomDensity, float *vol,
        float *extSource)
{
    int ir = blockIdx.x*Ny*Nz + blockIdx.y*Nz + threadIdx.x;

    totalSource[ir] = extSource[ir];
    for(unsigned int iie = highestEnergyGroup; iie < sinkGroup; iie++)
    {
        //int zidIndx = zoneId[ir];
        //         [#]  +=  [#]         *  [#/cm^2]                     *  [b]                                                                 *  [1/b-cm]       *  [cm^3]
        totalSource[ir] += CUDA_4PI_INV * scalarFlux[iie*Nx*Ny*Nz + ir] * scatxs2d[zoneId[ir]*groups*groups*pn + iie*groups*pn + sinkGroup*pn] * atomDensity[ir] * vol[ir];
    }

}

__global__ void clearSweepKernel(
        float *preFlux, float *tempFlux,
        int Nx, int Ny, int Nz)
{
    int ir = blockIdx.x*Ny*Nz + blockIdx.y*Nz + threadIdx.x;

    preFlux[ir] = tempFlux[ir];
    tempFlux[ir] = 0.0f;
}

/*
__global__ void isoDiffKernel(float *scalarFlux, float *prevScalarFlux, float *diffMatrix, int Nx, int Ny, int Nz)
{
    //int ir = blockIdx.x*Ny*Nz + blockIdx.y*Nz + threadIdx.x;
    int tid = blockIdx.x*

    for(unsigned int i = 0; i < Nz; i++)
    {
        diffMatrix[] = 5;
    }


    return;
}
*/

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
// This code is taken from the CUDA SDK (6_Advanced/reduction) and modified to perform the diff reduction
/*
template <unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(float *g_idata, float *g_odata, unsigned int n)
{
    float *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    float mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum = fmaxf(mySum, g_idata[i]);
        //mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum = fmaxf(mySum, g_idata[i+blockSize]);
            //mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = fmaxf(mySum, sdata[tid+256]);  //mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = fmaxf(mySum, sdata[tid+128]);  //mySum + sdata[tid + 128];
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = fmaxf(mySum, sdata[tid+64]);  //mySum + sdata[tid +  64];
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum = fmaxf(mySum, sdata[tid+32]);  //mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            mySum = fmaxf(mySum, __shfl_down(mySum, offset));  //
            //mySum += __shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = fmaxf(mySum, sdata[tid+32]);  //mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = fmaxf(mySum, sdata[tid+16]);  //mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = fmaxf(mySum, sdata[tid+8]);  //mySum + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = fmaxf(mySum, sdata[tid+4]);  //mySum + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = fmaxf(mySum, sdata[tid+2]);  //mySum + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = fmaxf(mySum, sdata[tid+1]);  //mySum + sdata[tid +  1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}
*/


