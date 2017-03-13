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
    int xIndxStart = blockIdx.x;
    int yIndxStart = blockIdx.y;
    int zIndxStart = threadIdx.x;

    printf("<<<%i, %i, %i>>>\n", xIndxStart, yIndxStart, zIndxStart);

    bool DEBUG_ = false;
    if(xIndxStart == 35 && yIndxStart == 0 && zIndxStart == 8)
    {
        DEBUG_ = true;
    }

    const unsigned short DIRECTION_X = 1;
    const unsigned short DIRECTION_Y = 2;
    const unsigned short DIRECTION_Z = 3;

    RAY_T tiny = 1.0E-35f;
    RAY_T huge = 1.0E35f;

    int ir0 = xIndxStart*Ny*Nz + yIndxStart*Nz + zIndxStart;

    float *meanFreePaths;
    cudaMalloc(&meanFreePaths, groups*sizeof(float));

    // This runs for a single voxel
    RAY_T x = xNodes[xIndxStart] + dx[xIndxStart]/2;
    RAY_T y = yNodes[yIndxStart] + dy[yIndxStart]/2;
    RAY_T z = zNodes[zIndxStart] + dz[zIndxStart]/2;

    if(DEBUG_)
    {
        printf("CUDA: x = %f", x);
        printf("CUDA: y = %f", y);
        printf("CUDA: z = %f", z);
    }

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
    while(!exhaustedRay)
    {
        int ir = xIndx*Ny*Nz + yIndx*Nz + zIndx;

        if(DEBUG_)
        {
            printf("CUDA: ir=%i", ir);
            printf("CUDA: ix=%i", xIndx);
            printf("CUDA: iy=%i", yIndx);
            printf("CUDA: iz=%i", zIndx);
        }

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

        //if(tmin < -3E-8)  // Include a little padding for hitting an edge
        //    qDebug() << "Reversed space!";

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

    } // End of while loop

    for(unsigned int ie = 0; ie < groups; ie++)
    {
        RAY_T flx = srcStrength[ie] * exp(-meanFreePaths[ie]) / (CUDA_4PI * srcToCellDist * srcToCellDist);
        uflux[ie*Nx*Ny*Nz + ir0] = static_cast<SOL_T>(flx);  //srcStrength * exp(-meanFreePaths[ie]) / (4 * M_PI * srcToCellDist * srcToCellDist);
    }
}

__global__ void isoSolKernel(float *prevdata, float *data, int nx, int ny)
{
    /*
    unsigned int groups = xs->groupCount();

    const unsigned short DIRECTION_X = 1;
    const unsigned short DIRECTION_Y = 2;
    const unsigned short DIRECTION_Z = 3;

    const RAY_T sx = static_cast<RAY_T>(params->sourceX);
    const RAY_T sy = static_cast<RAY_T>(params->sourceY);
    const RAY_T sz = static_cast<RAY_T>(params->sourceZ);

    std::vector<SOL_T> *uflux = new std::vector<SOL_T>;
    uflux->resize(groups * mesh->voxelCount());

    unsigned int ejmp = mesh->voxelCount();
    unsigned int xjmp = mesh->xjmp();
    unsigned int yjmp = mesh->yjmp();

    qDebug() << "Running raytracer";

    RAY_T tiny = 1.0E-35f;
    RAY_T huge = 1.0E35f;
    std::vector<RAY_T> meanFreePaths;
    meanFreePaths.resize(xs->groupCount());

    //                                  0  1  2  3  4  5  6  7  8  9  0  1  2  3  4  5  6  7  8
    //std::vector<RAY_T> srcStrength(groups, 0.0);  //{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    //srcStrength[srcStrength.size() - 2] = 1.0;
    std::vector<RAY_T> srcStrength(groups, 0.0);
    for(unsigned int i = 0; i < groups; i++)
        srcStrength[i] = params->spectraIntensity[i];

    if(sx < mesh->xNodes[0] || sy < mesh->yNodes[0] || sz < mesh->zNodes[0])
    {
        qCritical() << "Source is ouside the mesh region on the negative side";
    }

    if(sx > mesh->xNodes[mesh->xNodeCt-1] || sy > mesh->yNodes[mesh->yNodeCt-1] || sz > mesh->zNodes[mesh->zNodeCt-1])
    {
        qCritical() << "Source is ouside the mesh region on the positive side";
    }

    unsigned int srcIndxX = 0;
    unsigned int srcIndxY = 0;
    unsigned int srcIndxZ = 0;

    while(mesh->xNodes[srcIndxX+1] < sx)
        srcIndxX++;

    while(mesh->yNodes[srcIndxY+1] < sy)
        srcIndxY++;

    while(mesh->zNodes[srcIndxZ+1] < sz)
        srcIndxZ++;

    unsigned int totalMissedVoxels = 0;

    for(unsigned int zIndxStart = 0; zIndxStart < mesh->zElemCt; zIndxStart++)
        for(unsigned int yIndxStart = 0; yIndxStart < mesh->yElemCt; yIndxStart++)
            for(unsigned int xIndxStart = 0; xIndxStart < mesh->xElemCt; xIndxStart++)  // For every voxel
            {
                //qDebug() << "voxel " << xIndxStart << " " << yIndxStart << " " << zIndxStart;
                RAY_T x = mesh->xNodes[xIndxStart] + mesh->dx[xIndxStart]/2;
                RAY_T y = mesh->yNodes[yIndxStart] + mesh->dy[yIndxStart]/2;
                RAY_T z = mesh->zNodes[zIndxStart] + mesh->dz[zIndxStart]/2;

                if(xIndxStart == srcIndxX && yIndxStart == srcIndxY && zIndxStart == srcIndxZ)  // End condition
                {
                    RAY_T srcToCellDist = sqrt((x-sx)*(x-sx) + (y-sy)*(y-sy) + (z-sz)*(z-sz));
                    unsigned int zid = mesh->zoneId[xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart];
                    RAY_T xsval;
                    for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                    {
                        xsval = xs->m_tot1d[zid*groups + ie] * mesh->atomDensity[xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart];
                        (*uflux)[ie*ejmp + xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart] = srcStrength[ie] * exp(-xsval*srcToCellDist) / (4 * m_pi * srcToCellDist * srcToCellDist);
                    }
                    continue;
                }

                // Start raytracing through the geometry
                unsigned int xIndx = xIndxStart;
                unsigned int yIndx = yIndxStart;
                unsigned int zIndx = zIndxStart;

                RAY_T srcToCellX = sx - x;
                RAY_T srcToCellY = sy - y;
                RAY_T srcToCellZ = sz - z;

                RAY_T srcToCellDist = sqrt(srcToCellX*srcToCellX + srcToCellY*srcToCellY + srcToCellZ*srcToCellZ);

                RAY_T xcos = srcToCellX/srcToCellDist;  // Fraction of direction biased in x-direction, unitless
                RAY_T ycos = srcToCellY/srcToCellDist;
                RAY_T zcos = srcToCellZ/srcToCellDist;

                int xBoundIndx = (xcos >= 0 ? xIndx+1 : xIndx);
                int yBoundIndx = (ycos >= 0 ? yIndx+1 : yIndx);
                int zBoundIndx = (zcos >= 0 ? zIndx+1 : zIndx);

                // Clear the MPF array to zeros
                for(unsigned int i = 0; i < xs->groupCount(); i++)
                    meanFreePaths[i] = 0.0f;

                bool exhaustedRay = false;
                while(!exhaustedRay)
                {
                    // Determine the distance to cell boundaries
                    RAY_T tx = (fabs(xcos) < tiny ? huge : (mesh->xNodes[xBoundIndx] - x)/xcos);  // Distance traveled [cm] when next cell is
                    RAY_T ty = (fabs(ycos) < tiny ? huge : (mesh->yNodes[yBoundIndx] - y)/ycos);  //   entered traveling in x direction
                    RAY_T tz = (fabs(zcos) < tiny ? huge : (mesh->zNodes[zBoundIndx] - z)/zcos);

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

                    if(tmin < -3E-8)  // Include a little padding for hitting an edge
                        qDebug() << "Reversed space!";

                    // Update mpf array
                    unsigned int zid = mesh->zoneId[xIndx*xjmp + yIndx*yjmp + zIndx];
                    for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                    {
                        //                   [cm] * [b] * [atom/b-cm]
                        meanFreePaths[ie] += tmin * xs->m_tot1d[zid*groups + ie] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx];
                    }

                    // Update cell indices and positions
                    if(dirHitFirst == DIRECTION_X) // x direction
                    {
                        x = mesh->xNodes[xBoundIndx];
                        y += tmin*ycos;
                        z += tmin*zcos;
                        if(xcos >= 0)
                        {
                            xIndx++;
                            xBoundIndx++;
                            if(xIndx > srcIndxX)
                            {
                                totalMissedVoxels++;
                                qCritical() << "Missed the target x+ bound: Started at " << xIndxStart << " " + yIndxStart << " " << zIndxStart << " Aiming for " << srcIndxX << " " << srcIndxY << " " << srcIndxZ << " and hit " << xIndx << " " << yIndx << " " << zIndx << "Missed voxel # " << totalMissedVoxels;
                                exhaustedRay = true;
                            }
                        }
                        else
                        {
                            xIndx--;
                            xBoundIndx--;
                            if(xIndx < srcIndxX)
                            {
                                totalMissedVoxels++;
                                qCritical() << "Missed the target x- bound: Started at " << xIndxStart << " " + yIndxStart << " " << zIndxStart << " Aiming for " << srcIndxX << " " << srcIndxY << " " << srcIndxZ << " and hit " << xIndx << " " << yIndx << " " << zIndx << "Missed voxel # " << totalMissedVoxels;
                                exhaustedRay = true;
                            }
                        }
                    }
                    else if(dirHitFirst == DIRECTION_Y) // y direction
                    {
                        x += tmin*xcos;
                        y = mesh->yNodes[yBoundIndx];
                        z += tmin*zcos;
                        if(ycos >= 0)
                        {
                            yIndx++;
                            yBoundIndx++;
                            if(yIndx > srcIndxY)
                            {
                                totalMissedVoxels++;
                                qCritical() << "Missed the target y+ bound: Started at " << xIndxStart << " " + yIndxStart << " " << zIndxStart << " Aiming for " << srcIndxX << " " << srcIndxY << " " << srcIndxZ << " and hit " << xIndx << " " << yIndx << " " << zIndx << "Missed voxel # " << totalMissedVoxels;
                                exhaustedRay = true;
                            }
                        }
                        else
                        {
                            yIndx--;
                            yBoundIndx--;
                            if(yIndx < srcIndxY)
                            {
                                totalMissedVoxels++;
                                qCritical() << "Missed the target y- bound: Started at " << xIndxStart << " " + yIndxStart << " " << zIndxStart << " Aiming for " << srcIndxX << " " << srcIndxY << " " << srcIndxZ << " and hit " << xIndx << " " << yIndx << " " << zIndx << "Missed voxel # " << totalMissedVoxels;
                                exhaustedRay = true;
                            }
                        }
                    }
                    else if(dirHitFirst == DIRECTION_Z) // z direction
                    {
                        x += tmin*xcos;
                        y += tmin*ycos;
                        z = mesh->zNodes[zBoundIndx];
                        if(zcos >= 0)
                        {
                            zIndx++;
                            zBoundIndx++;
                            if(zIndx > srcIndxZ)
                            {
                                totalMissedVoxels++;
                                qCritical() << "Missed the target z+ bound: Started at " << xIndxStart << " " + yIndxStart << " " << zIndxStart << " Aiming for " << srcIndxX << " " << srcIndxY << " " << srcIndxZ << " and hit " << xIndx << " " << yIndx << " " << zIndx << "Missed voxel # " << totalMissedVoxels;
                                exhaustedRay = true;
                            }
                        }
                        else
                        {
                            zIndx--;
                            zBoundIndx--;
                            if(zIndx < srcIndxZ)
                            {
                                totalMissedVoxels++;
                                qCritical() << "Missed the target z- bound: Started at " << xIndxStart << " " + yIndxStart << " " << zIndxStart << " Aiming for " << srcIndxX << " " << srcIndxY << " " << srcIndxZ << " and hit " << xIndx << " " << yIndx << " " << zIndx << "Missed voxel # " << totalMissedVoxels;
                                exhaustedRay = true;
                            }
                        }
                    }

                    if((xIndx == srcIndxX && yIndx == srcIndxY && zIndx == srcIndxZ) || exhaustedRay)
                    {
                        RAY_T finalDist = sqrt((x-sx)*(x-sx) + (y-sy)*(y-sy) + (z-sz)*(z-sz));

                        for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                        {
                            //       [#]       = [cm] * [b] * [1/cm-b]
                            meanFreePaths[ie] += finalDist * xs->m_tot1d[zid*groups + ie] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx];
                        }

                        exhaustedRay = true;
                    }

                } // End of while loop

                for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                {
                    RAY_T flx = srcStrength[ie] * exp(-meanFreePaths[ie]) / (m_4pi * srcToCellDist * srcToCellDist);

                    if(flx < 0)
                        qDebug() << "solver.cpp: (291): Negative?";

                    if(flx > 1E6)
                        qDebug() << "solver.cpp: (294): Too big!";

                    (*uflux)[ie*ejmp + xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart] = static_cast<SOL_T>(flx);  //srcStrength * exp(-meanFreePaths[ie]) / (4 * M_PI * srcToCellDist * srcToCellDist);
                }

            } // End of each voxel

    return uflux;

    */
}
