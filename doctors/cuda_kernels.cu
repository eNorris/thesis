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
        int sweepNumber, int voxThisSubSweep)
{

    int i = blockIdx.x*Ny*Nz + blockIdx.y*Nz + threadIdx.x;

    if(i >= voxThisSubSweep)
        return;

    int level = static_cast<int>((1 + sqrtf(1 + 8*i))/2.0);
    int prior = level * (level + 1) / 2;

    int ix = sweepNumber - level;
    int iy = 5; // TODO
    int iz = 5; // TODO
    int ir = ix*Ny*Nz + iy*Nz + iz;

    float influxX, influxY, influxZ;

    int zid = zoneId[ir];  // Get the zone id of this element

    // Handle the x influx
    if(mu[iang] >= 0)                                       // Approach x = 0 -> xMesh
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
    if(xi[iang] >= 0)                                       // Approach y = 0 -> yMesh
    {
        if(iy == 0)
            influxY = 0.0f;
        else
            influxY = outboundFluxY[ir - Nz];
    }
    else                                                          // Approach y = yMesh-1 -> 0
    {
        if(iy == (signed) Ny - 1)
            influxY = 0.0f;
        else
            influxY = outboundFluxY[ir + Nz];
    }

    // Handle the z influx
    if(eta[iang] >= 0)
    {
        if(iz == 0)
            influxZ = 0.0f;
        else
            influxZ = outboundFluxZ[ir - 1];
    }
    else
    {
        if(iz == (signed) Nz - 1)
            influxZ = 0.0f;
        else
            influxZ = outboundFluxZ[ir + 1];
    }

    //SOL_T inscatter = CUDA_4PI_INV*scalarFlux[ie*Nx*Ny*Nz + ir] * scatxs2d(zid, ie, ie, 0) * atomDensity[ir] * vol[ir];
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
    for(unsigned int iep = highestEnergyGroup; iep <= sinkGroup; iep++)  // Sink energy
        extSource[sinkGroup*voxels + ir] += uFlux[iep*voxels + ir] * vol[ir] * scatxs2d[zoneId[ir]*groups*groups*pn + iep*groups*pn + sinkGroup*pn] * atomDensity[ir];
}
__global__ void zeroKernel(int Nx, int Ny, int Nz, float *ptr)
{
    int ir = blockIdx.x*Ny*Nz + blockIdx.y*Nz + threadIdx.x;
    ptr[ir] = 0.0f;
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

    for(unsigned int iie = highestEnergyGroup; iie < sinkGroup; iie++)
    {
        //int zidIndx = zoneId[ir];
        //         [#]  +=  [#]         *  [#/cm^2]                     *  [b]                                                                 *  [1/b-cm]       *  [cm^3]
        totalSource[ir] += CUDA_4PI_INV * scalarFlux[iie*Nx*Ny*Nz + ir] * scatxs2d[zoneId[ir]*groups*groups*pn + iie*groups*pn + sinkGroup*pn] * atomDensity[ir] * vol[ir];
    }
    totalSource[ir] += extSource[sinkGroup*Nx*Ny*Nz + ir];
}

__global__ void clearSweepKernel(
        float *preFlux, float *tempFlux,
        int Nx, int Ny, int Nz)
{
    int ir = blockIdx.x*Ny*Nz + blockIdx.y*Nz + threadIdx.x;

    preFlux[ir] = tempFlux[ir];
    tempFlux[ir] = 0.0f;

    //preFlux = tempFlux;  // Store flux for previous iteration


    // Clear for a new sweep
    //for(unsigned int i = 0; i < tempFlux.size(); i++)
    //    tempFlux[i] = 0;
}

