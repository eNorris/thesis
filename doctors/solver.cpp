#include "solver.h"

#include <iostream>
#include <ctime>

#include <QDebug>
#include <QThread>

#include "quadrature.h"
#include "mesh.h"
#include "xsection.h"
#include "outwriter.h"
#include "gui/outputdialog.h"
#include "legendre.h"

Solver::Solver(QObject *parent) : QObject(parent)
{

}

Solver::~Solver()
{

}

std::vector<RAY_T> *Solver::basicRaytrace(const Quadrature *quad, const Mesh *mesh, const XSection *xs, RAY_T sx, RAY_T sy, RAY_T sz)
{
    int groups = xs->groupCount();

    const unsigned short DIRECTION_X = 1;
    const unsigned short DIRECTION_Y = 2;
    const unsigned short DIRECTION_Z = 3;

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
    std::vector<RAY_T> srcStrength(groups, 0.0);  //{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    srcStrength[srcStrength.size() - 2] = 1.0;

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


    //unsigned int srcIndxX = int(sx / mesh->dx[0]);  // TODO: This will only work for evenly spaced grids
    //unsigned int srcIndxY = int(sy / mesh->dy[0]);
    //unsigned int srcIndxZ = int(sz / mesh->dz[0]);

    unsigned int totalMissedVoxels = 0;

    for(unsigned int zIndxStart = 0; zIndxStart < mesh->zElemCt; zIndxStart++)
        for(unsigned int yIndxStart = 0; yIndxStart < mesh->yElemCt; yIndxStart++)
            for(unsigned int xIndxStart = 0; xIndxStart < mesh->xElemCt; xIndxStart++)  // For every voxel
            {
                //qDebug() << "voxel " << xIndxStart << " " << yIndxStart << " " << zIndxStart;
                RAY_T x = mesh->xNodes[xIndxStart] + mesh->dx[xIndxStart]/2;
                RAY_T y = mesh->yNodes[yIndxStart] + mesh->dy[yIndxStart]/2;
                RAY_T z = mesh->zNodes[zIndxStart] + mesh->dz[zIndxStart]/2;

                //std::vector<float> tmpdistv;
                //std::vector<float> tmpxsv;
                //std::vector<float> mfpv;

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
                    //qDebug() << x << " " << y << " "  << z << " " << xIndx << " " << yIndx << " " << zIndx;
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
                    //tmpdistv.push_back(tmin);
                    //tmpxsv.push_back(xs->m_tot1d[zid*groups + 18] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx]);
                    //float gain = tmin * xs->m_tot1d[zid*groups + 17] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx];
                    //mfpv.push_back(gain);

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
}

void Solver::raytraceIso(const Quadrature *quad, const Mesh *mesh, const XSection *xs)
{
    std::clock_t startMoment = std::clock();

    //unsigned int groups = xs->groupCount();
    RAY_T sx = 25.3906f;
    RAY_T sy = 50.0f - 46.4844f;
    RAY_T sz = 6.8906f;

    //unsigned int ejmp = mesh->voxelCount();
    //unsigned int xjmp = mesh->xjmp();
    //unsigned int yjmp = mesh->yjmp();

    std::vector<SOL_T> *uflux = basicRaytrace(quad, mesh, xs, sx, sy, sz);

    qDebug() << "Time to complete raytracer: " << (std::clock() - startMoment)/(double)(CLOCKS_PER_SEC/1000) << " ms";

    emit signalRaytracerFinished(uflux);
    emit signalNewRaytracerIteration(uflux);
}


void Solver::gsSolverIso(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const std::vector<SOL_T> *uFlux)
{

    std::clock_t startTime = std::clock();

    const int maxIterations = 25;
    const SOL_T epsilon = 0.01f;


    //std::vector<SOL_T> angularFlux(xs->groupCount() * quad->angleCount() * mesh->voxelCount());
    std::vector<SOL_T> *scalarFlux = new std::vector<SOL_T>(xs->groupCount() * mesh->voxelCount(), 0.0f);
    std::vector<SOL_T> tempFlux(mesh->voxelCount(), 0.0f);
    std::vector<SOL_T> preFlux(mesh->voxelCount(), -100.0f);
    std::vector<SOL_T> totalSource(mesh->voxelCount(), -100.0f);
    std::vector<SOL_T> outboundFluxX(mesh->voxelCount(), -100.0f);
    std::vector<SOL_T> outboundFluxY(mesh->voxelCount(), -100.0f);
    std::vector<SOL_T> outboundFluxZ(mesh->voxelCount(), -100.0f);
    std::vector<SOL_T> extSource(xs->groupCount() * mesh->voxelCount(), 0.0f);

    std::vector<SOL_T> errMaxList;
    std::vector<std::vector<SOL_T> > errList;
    std::vector<int> converganceIters;
    std::vector<SOL_T> converganceTracker;

    errMaxList.resize(xs->groupCount());
    errList.resize(xs->groupCount());
    converganceIters.resize(xs->groupCount());
    converganceTracker.resize(xs->groupCount());

    SOL_T influxX = 0.0f;
    SOL_T influxY = 0.0f;
    SOL_T influxZ = 0.0f;

    //int ejmp = mesh->voxelCount() * quad->angleCount();
    //int ajmp = mesh->voxelCount();
    int xjmp = mesh->xjmp();
    int yjmp = mesh->yjmp();

    //bool downscatterFlag = false;

    bool noDownscatterYet = true;
    unsigned int highestEnergy = 0;

    while(noDownscatterYet)
    {
        SOL_T dmax = 0.0;
        unsigned int vc = mesh->voxelCount();
        for(unsigned int ira = 0; ira < vc; ira++)
        {
            dmax = (dmax > (*uFlux)[highestEnergy*vc + ira]) ? dmax : (*uFlux)[highestEnergy*vc + ira];
        }
        if(dmax <= 0.0)
        {
            qDebug() << "No external source or downscatter, skipping energy group " << highestEnergy;
            highestEnergy++;
        }
        else
        {
            noDownscatterYet = false;
        }
    }

    if(uFlux != NULL)
    {
        qDebug() << "Loading uncollided flux into external source";
        // If there is an uncollided flux provided, use it, otherwise, calculate the external source
        //unsigned int xx= xs->groupCount();
        for(unsigned int ie = highestEnergy; ie < xs->groupCount(); ie++)  // Sink energy
            for(unsigned int ri = 0; ri < mesh->voxelCount(); ri++)
                for(unsigned int iep = highestEnergy; iep <= ie; iep++) // Source energy
                    //                               [#]   =                        [#/cm^2]      * [cm^3]        *  [b]                               * [1/b-cm]
                    extSource[ie*mesh->voxelCount() + ri] += (*uFlux)[iep*mesh->voxelCount() + ri] * mesh->vol[ri] * xs->scatxs2d(mesh->zoneId[ri], iep, ie, 0) * mesh->atomDensity[ri];

        OutWriter::writeArray("externalSrc.dat", extSource);
    }
    else
    {
        qDebug() << "Building external source";
        int srcIndxE = xs->groupCount() - 1;
        int srcIndxX = 32;
        int srcIndxY = 4;  //mesh->yElemCt/2;
        int srcIndxZ = 8;
        //                                                                              [#] = [#]
        extSource[srcIndxE * mesh->voxelCount() + srcIndxX*xjmp + srcIndxY*yjmp + srcIndxZ] = 1.0;
    }

    qDebug() << "Solving " << mesh->voxelCount() * quad->angleCount() * xs->groupCount() << " elements in phase space";


    for(unsigned int ie = highestEnergy; ie < xs->groupCount(); ie++)  // for every energy group
    {
        int iterNum = 1;
        SOL_T maxDiff = 1.0;

        for(unsigned int i = 0; i < totalSource.size(); i++)
            totalSource[i] = 0;

        // Calculate the down-scattering source
        for(unsigned int iie = highestEnergy; iie < ie; iie++)
            for(unsigned int ir = 0; ir < mesh->voxelCount(); ir++)
            {
                int zidIndx = mesh->zoneId[ir];
                //         [#]    +=  [#]          *      [#/cm^2]                               * [b]                          * [1/b-cm]                * [cm^3]
                totalSource[ir] += m_4pi_inv*(*scalarFlux)[iie*mesh->voxelCount() + ir] * xs->scatxs2d(zidIndx, iie, ie, 0) * mesh->atomDensity[ir] * mesh->vol[ir]; //xsref(ie-1, zidIndx, 0, iie));
            }

        // Add the external source
        for(unsigned int ri = 0; ri < mesh->voxelCount(); ri++)
        {
            //  [#]         +=  [#]
            totalSource[ri] += extSource[ie*mesh->voxelCount() + ri];
        }

        while(iterNum <= maxIterations && maxDiff > epsilon)  // while not converged
        {
            //qDebug() << "Iteration #" << iterNum;

            preFlux = tempFlux;  // Store flux for previous iteration


            // Clear for a new sweep
            for(unsigned int i = 0; i < tempFlux.size(); i++)
                tempFlux[i] = 0;

            for(unsigned int iang = 0; iang < quad->angleCount(); iang++)  // for every angle
            {
                //qDebug() << "Angle #" << iang;

                // Find the correct direction to sweep
                int izStart = 0;                  // Sweep start index
                int diz = 1;                      // Sweep direction
                if(quad->eta[iang] < 0)           // Condition to sweep backward
                {
                    izStart = mesh->zElemCt - 1;  // Start at the far end
                    diz = -1;                     // Sweep toward zero
                }

                int iyStart = 0;
                int diy = 1;
                if(quad->zi[iang] < 0)
                {
                    iyStart = mesh->yElemCt - 1;
                    diy = -1;
                }

                int ixStart = 0;
                int dix = 1;
                if(quad->mu[iang] < 0)
                {
                    ixStart = mesh->xElemCt - 1;
                    dix = -1;
                }

                int iz = izStart;
                while(iz < (signed) mesh->zElemCt && iz >= 0)
                {
                    int iy = iyStart;
                    while(iy < (signed) mesh->yElemCt && iy >= 0)
                    {
                        int ix = ixStart;
                        while(ix < (signed) mesh->xElemCt && ix >= 0)  // for every mesh element in the proper order
                        {
                            int zid = mesh->zoneId[ix*xjmp + iy*yjmp + iz];  // Get the zone id of this element

                            // Handle the x influx
                            if(quad->mu[iang] >= 0)                                       // Approach x = 0 -> xMesh
                            {
                                if(ix == 0)                                               // If this is a boundary cell
                                    influxX = 0.0f;                                       // then the in-flux is zero
                                else                                                      // otherwise
                                    influxX = outboundFluxX[(ix-1)*xjmp + iy*yjmp + iz];  // the in-flux is the out-flux from the previous cell
                            }
                            else                                                          // Approach x = xMesh-1 -> 0
                            {
                                if(ix == (signed) mesh->xElemCt-1)
                                    influxX = 0.0f;
                                else
                                    influxX = outboundFluxX[(ix+1)*xjmp + iy*yjmp + iz];
                            }

                            // Handle the y influx
                            if(quad->zi[iang] >= 0)                                       // Approach y = 0 -> yMesh
                            {
                                if(iy == 0)
                                    influxY = 0.0f;
                                else
                                    influxY = outboundFluxY[ix*xjmp + (iy-1)*yjmp + iz];
                            }
                            else                                                          // Approach y = yMesh-1 -> 0
                            {
                                if(iy == (signed) mesh->yElemCt-1)
                                    influxY = 0.0f;
                                else
                                    influxY = outboundFluxY[ix*xjmp + (iy+1)*yjmp + iz];
                            }

                            // Handle the z influx
                            if(quad->eta[iang] >= 0)
                            {
                                if(iz == 0)
                                    influxZ = 0.0f;
                                else
                                    influxZ = outboundFluxZ[ix*xjmp + iy*yjmp + iz-1];
                            }
                            else
                            {
                                if(iz == (signed) mesh->zElemCt-1)
                                    influxZ = 0.0f;
                                else
                                    influxZ = outboundFluxZ[ix*xjmp + iy*yjmp + iz+1];
                            }

                            SOL_T inscatter = m_4pi_inv*(*scalarFlux)[ie*mesh->voxelCount() + ix*xjmp+iy*yjmp+iz] * xs->scatxs2d(zid, ie, ie, 0) * mesh->atomDensity[ix*xjmp+iy*yjmp+iz] * mesh->vol[ix*xjmp+iy*yjmp+iz];

                            SOL_T numer = totalSource[ix*xjmp+iy*yjmp+iz] +  inscatter +                                                                                            // [#]
                                    mesh->Ayz[ie*quad->angleCount()*mesh->yElemCt*mesh->zElemCt + iang*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz] * influxX +  // [cm^2 * #/cm^2]  The 2x is already factored in
                                    mesh->Axz[ie*quad->angleCount()*mesh->xElemCt*mesh->zElemCt + iang*mesh->xElemCt*mesh->zElemCt + ix*mesh->zElemCt + iz] * influxY +
                                    mesh->Axy[ie*quad->angleCount()*mesh->xElemCt*mesh->yElemCt + iang*mesh->xElemCt*mesh->yElemCt + ix*mesh->yElemCt + iy] * influxZ;
                            SOL_T denom = mesh->vol[ix*xjmp+iy*yjmp+iz]*xs->totXs1d(zid, ie)*mesh->atomDensity[ix*xjmp + iy*yjmp + iz] +                               // [cm^3] * [b] * [1/b-cm]
                                    mesh->Ayz[ie*quad->angleCount()*mesh->yElemCt*mesh->zElemCt + iang*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz] +            // [cm^2]
                                    mesh->Axz[ie*quad->angleCount()*mesh->xElemCt*mesh->zElemCt + iang*mesh->xElemCt*mesh->zElemCt + ix*mesh->zElemCt + iz] +
                                    mesh->Axy[ie*quad->angleCount()*mesh->xElemCt*mesh->yElemCt + iang*mesh->xElemCt*mesh->yElemCt + ix*mesh->yElemCt + iy];

                            //   [#/cm^2] = [#]  / [cm^2]
                            SOL_T angFlux = numer/denom;

                            if(std::isnan(angFlux))
                            {
                                qDebug() << "Found a nan!";
                                qDebug() << "Vol = " << mesh->vol[ix*xjmp+iy*yjmp+iz];
                                qDebug() << "xs = " << xs->totXs1d(zid, ie);
                                qDebug() << "Ayz = " << mesh->Ayz[iang*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz];
                                qDebug() << "Axz = " << mesh->Axz[iang*mesh->xElemCt*mesh->zElemCt + ix*mesh->zElemCt + iz];
                                qDebug() << "Axy = " << mesh->Axy[iang*mesh->xElemCt*mesh->yElemCt + ix*mesh->yElemCt + iy];
                            }

                            //angularFlux[ie*ejmp + iang*ajmp + ix*xjmp + iy*yjmp + iz] = angFlux;

                            outboundFluxX[ix*xjmp + iy*yjmp + iz] = 2*angFlux - influxX;
                            outboundFluxY[ix*xjmp + iy*yjmp + iz] = 2*angFlux - influxY;
                            outboundFluxZ[ix*xjmp + iy*yjmp + iz] = 2*angFlux - influxZ;

                            // Sum all the angular fluxes
                            tempFlux[ix*xjmp + iy*yjmp + iz] += quad->wt[iang]*angFlux;

                            ix += dix;
                        } // end of for ix

                        iy += diy;
                    } // end of for iy

                    iz += diz;
                } // end of for iz

                for(unsigned int i = 0; i < tempFlux.size(); i++)
                {
                    (*scalarFlux)[ie*mesh->voxelCount() + i] = tempFlux[i];
                }

                emit signalNewSolverIteration(scalarFlux);

                unsigned int xTracked = mesh->xElemCt/2;
                unsigned int yTracked = mesh->yElemCt/2;
                unsigned int zTracked = mesh->zElemCt/2;
                converganceTracker.push_back((*scalarFlux)[ie*mesh->voxelCount() + xTracked*xjmp + yTracked*yjmp + zTracked]);

            } // end of all angles

            maxDiff = -1.0E35f;
            for(unsigned int i = 0; i < tempFlux.size(); i++)
            {
                //float z = qAbs((tempFlux[i] - preFlux[i])/tempFlux[i]);
                maxDiff = qMax(maxDiff, qAbs((tempFlux[i] - preFlux[i])/tempFlux[i]));

                if(std::isnan(maxDiff))
                    qDebug() << "Found a diff nan!";
            }
            qDebug() << "Max diff = " << maxDiff;

            errList[ie].push_back(maxDiff);
            errMaxList[ie] = maxDiff;
            converganceIters[ie] = iterNum;

            for(unsigned int i = 0; i < tempFlux.size(); i++)
            {
                (*scalarFlux)[ie*mesh->voxelCount() + i] = tempFlux[i];
            }

            iterNum++;
        } // end not converged
    }  // end each energy group

    qDebug() << "Time to complete: " << (std::clock() - startTime)/(double)(CLOCKS_PER_SEC/1000) << " ms";

    //qDebug() << "Convergance of 128, 128, 32:";
    //for(unsigned int i = 0; i < converganceTracker.size(); i++)\
    //{
    //    qDebug() << i << "\t" << converganceTracker[i];
    //}
    //qDebug() << "";

    for(unsigned int i = 0; i < errList.size(); i++)
    {
        std::cout << "Group: " << i << "   maxDiff: " << errMaxList[i] << "   Iterations: " << converganceIters[i] << '\n';
        for(unsigned int j = 0; j < errList[i].size(); j++)
            std::cout << errList[i][j] << '\t';
        std::cout << std::endl;
    }

    emit signalNewSolverIteration(scalarFlux);
    emit signalSolverFinished(scalarFlux);
}


// ////////////////////////////////////////////////////////////////////////////////////////////// //
//                           Anisotropic versions of the above solvers                            //
// ////////////////////////////////////////////////////////////////////////////////////////////// //

void Solver::raytraceLegendre(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const unsigned int pn)
{
    std::clock_t startMoment = std::clock();

    unsigned int groups = xs->groupCount();

    RAY_T sx = 25.3906f;
    RAY_T sy = 50.0f - 46.4844f;
    RAY_T sz = 6.8906f;

    unsigned int ejmp = mesh->voxelCount();
    unsigned int xjmp = mesh->xjmp();
    unsigned int yjmp = mesh->yjmp();

    std::vector<SOL_T> *uflux = basicRaytrace(quad, mesh, xs, sx, sy, sz);

    qDebug() << "Time to complete raytracer: " << (std::clock() - startMoment)/(double)(CLOCKS_PER_SEC/1000) << " ms";

    std::vector<RAY_T> *ufluxAng = new std::vector<RAY_T>;
    ufluxAng->resize(groups * quad->angleCount() * mesh->voxelCount(), 0.0f);

    int eajmp = quad->angleCount() * mesh->voxelCount();
    int aajmp = mesh->voxelCount();
    int xajmp = mesh->yElemCt * mesh->zElemCt;
    int yajmp = mesh->zElemCt;

    for(unsigned int ie = 0; ie < groups; ie++)
        for(unsigned int iz = 0; iz < mesh->zElemCt; iz++)
            for(unsigned int iy = 0; iy < mesh->yElemCt; iy++)
                for(unsigned int ix = 0; ix < mesh->xElemCt; ix++)  // For every voxel
                {
                    float x = mesh->xNodes[ix] + mesh->dx[ix]/2;
                    float y = mesh->yNodes[iy] + mesh->dy[iy]/2;
                    float z = mesh->zNodes[iz] + mesh->dz[iz]/2;

                    float deltaX = x - sx;
                    float deltaY = y - sy;
                    float deltaZ = z - sz;

                    // normalize to unit vector
                    float mag = sqrt(deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ);
                    deltaX /= mag;
                    deltaY /= mag;
                    deltaZ /= mag;

                    unsigned int bestAngIndx = 0;
                    float bestCosT = -1.0f;

                    for(unsigned int ia = 0; ia < quad->angleCount(); ia++)
                    {
                        float cosT = quad->mu[ia] * deltaX + quad->eta[ia] * deltaY + quad->zi[ia] * deltaZ;
                        if(cosT > bestCosT)
                        {
                            bestCosT = cosT;
                            bestAngIndx = ia;
                        }
                    }

                    (*ufluxAng)[ie*eajmp + bestAngIndx*aajmp + ix*xajmp + iy*yajmp + iz] = (*uflux)[ie*ejmp + ix*xjmp + iy*yjmp + iz];
                }

    qDebug() << "Raytracer + anisotropic mapping completed in " << (std::clock() - startMoment)/(double)(CLOCKS_PER_SEC/1000) << " ms";

    emit signalRaytracerFinished(ufluxAng);
    emit signalNewRaytracerIteration(uflux);
}


void Solver::gsSolverLegendre(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const unsigned int pn, const std::vector<RAY_T> *uFlux)
{
    // Do some input checks
    if(pn > 10)
    {
        qDebug() << "pn check failed, pn = " << pn;
        return;
    }

    std::clock_t startTime = std::clock();

    const int maxIterations = 25;
    const SOL_T epsilon = static_cast<SOL_T>(0.01);
    Legendre legendre;
    legendre.precompute(quad, pn);


    std::vector<SOL_T> *angularFlux = new std::vector<SOL_T>(xs->groupCount() * quad->angleCount() * mesh->voxelCount());
    std::vector<SOL_T> *scalarFlux = new std::vector<SOL_T>(xs->groupCount() * mesh->voxelCount(), 0.0f);
    std::vector<SOL_T> tempFlux(mesh->voxelCount());
    std::vector<SOL_T> preFlux(mesh->voxelCount(), -100.0f);
    std::vector<SOL_T> outboundFluxX(mesh->voxelCount(), -100.0f);
    std::vector<SOL_T> outboundFluxY(mesh->voxelCount(), -100.0f);
    std::vector<SOL_T> outboundFluxZ(mesh->voxelCount(), -100.0f);
    std::vector<SOL_T> extSource(xs->groupCount() * quad->angleCount() * mesh->voxelCount(), 0.0f);
    std::vector<SOL_T> totalSource(quad->angleCount() * mesh->voxelCount(), 0.0f);

    std::vector<SOL_T> errMaxList;
    std::vector<std::vector<SOL_T> > errList;
    std::vector<int> converganceIters;
    std::vector<SOL_T> converganceTracker;

    errMaxList.resize(xs->groupCount());
    errList.resize(xs->groupCount());
    converganceIters.resize(xs->groupCount());
    converganceTracker.resize(xs->groupCount());

    //const XSection &xsref = *xs;

    SOL_T influxX = 0.0f;
    SOL_T influxY = 0.0f;
    SOL_T influxZ = 0.0f;

    int ejmp = mesh->voxelCount() * quad->angleCount();
    int ajmp = mesh->voxelCount();
    int xjmp = mesh->xjmp();
    int yjmp = mesh->yjmp();

    bool noDownscatterYet = true;
    unsigned int highestEnergy = 0;

    while(noDownscatterYet)
    {
        SOL_T dmax = 0.0;
        unsigned int vc = mesh->voxelCount() * quad->angleCount();
        for(unsigned int ira = 0; ira < vc; ira++)
        {
            dmax = (dmax > (*uFlux)[highestEnergy*vc + ira]) ? dmax : (*uFlux)[highestEnergy*vc + ira];
        }
        if(dmax <= 0.0)
        {
            qDebug() << "No external source or downscatter, skipping energy group " << highestEnergy;
            highestEnergy++;
        }
        else
        {
            noDownscatterYet = false;
        }
    }

    if(uFlux != NULL)
    {
        qDebug() << "Computing 1st collision source";
        for(unsigned int ie = highestEnergy; ie < xs->groupCount(); ie++)
            for(unsigned int ia = 0; ia < quad->angleCount(); ia++)
                for(unsigned int ir = 0; ir < mesh->voxelCount(); ir++)
                {
                    float firstColSrc = 0.0f;
                    // TODO - should the equality condition be there?
                    for(unsigned int iep = highestEnergy; iep <= ie; iep++)  // For every higher energy that can downscatter
                        for(unsigned int il = 0; il <= pn; il++)  // For every Legendre expansion coeff
                        {
                            // The 2l+1 term is already accounted for in the XS
                            //float legendre_coeff = (2*l + 1) / M_4PI * xs->scatxs2d(mesh->zoneId[ri], epi, ei, l);  // [b]
                            SOL_T legendre_coeff = m_4pi_inv * xs->scatxs2d(mesh->zoneId[ir], iep, ie, il);  // [b]
                            SOL_T integral = static_cast<SOL_T>(0.0);
                            for(unsigned int iap = 0; iap < quad->angleCount(); iap++) // For every angle
                                integral += legendre.table(iap, ia, il) * (*uFlux)[iep*ejmp + iap*ajmp + ir] * quad->wt[iap];
                            // [b/cm^2]  = [b]  * [1/cm^2]
                            firstColSrc += legendre_coeff * integral;
                        }

                    //                          [#]   =    [b/cm^2]      * [cm^3]          * [1/b-cm]
                    extSource[ie*ejmp + ia*ajmp + ir] = firstColSrc * mesh->vol[ir] * mesh->atomDensity[ir];  //(*uFlux)[ei*ejmp + ai*ajmp + ri] * mesh->vol[ri] * xs->scatXs1d(mesh->zoneId[ri], ei) * mesh->atomDensity[ri];
                }

        qDebug() << "Finished 1st collision source computation";

        OutWriter::writeArray("externalSrc.dat", extSource);
    }
    else
    {
        qWarning() << "Building external source is illegal in anisotropic case!";

        //       [#] = [#]
        extSource[0] = 1.0;
    }

    qDebug() << "Solving " << mesh->voxelCount() * quad->angleCount() * xs->groupCount() << " elements in phase space";

    for(unsigned int ie = highestEnergy; ie < xs->groupCount(); ie++)  // for every energy group
    {
        qDebug() << "Energy group #" << ie;

        int iterNum = 1;
        SOL_T maxDiff = 1.0;

        for(unsigned int i = 0; i < totalSource.size(); i++)
            totalSource[i] = 0;

        // Compute the downscatter
        for(unsigned int iep = highestEnergy; iep < ie; iep++)
            for(unsigned int ia = 0; ia < quad->angleCount(); ia++)
                for(unsigned int iap = 0; iap < quad->angleCount(); iap++)
                    for(unsigned int ir = 0; ir < mesh->voxelCount(); ir++)
                        for(unsigned int il = 0; il <= pn; il++)
                        {
                            unsigned int zid = mesh->zoneId[ir];
                            totalSource[ia*mesh->voxelCount() + ir] += m_4pi_inv*legendre.table(ia, iap, il) *
                                    xs->scatxs2d(zid, iep, ie, il) *
                                    (*angularFlux)[iep*ejmp + iap*ajmp + ir] *
                                    quad->wt[iap] *
                                    mesh->atomDensity[ir] * mesh->vol[ir];
                        }

        for(unsigned int i = 0; i < quad->angleCount() * mesh->voxelCount(); i++)
            totalSource[i] += extSource[ie*quad->angleCount()*mesh->voxelCount() + i];

        while(iterNum <= maxIterations && maxDiff > epsilon)  // while not converged
        {
            //qDebug() << "Iteration #" << iterNum;

            preFlux = tempFlux;  // Store flux from previous iteration for convergance eval

            // Clear for a new sweep
            for(unsigned int i = 0; i < tempFlux.size(); i++)
                tempFlux[i] = 0;

            for(unsigned int ia = 0; ia < quad->angleCount(); ia++)  // for every angle
            {
                //qDebug() << "Angle #" << iang;

                // Find the correct direction to sweep
                int izStart = 0;                  // Sweep start index
                int diz = 1;                      // Sweep direction
                if(quad->eta[ia] < 0)           // Condition to sweep backward
                {
                    izStart = mesh->zElemCt - 1;  // Start at the far end
                    diz = -1;                     // Sweep toward zero
                }

                int iyStart = 0;
                int diy = 1;
                if(quad->zi[ia] < 0)
                {
                    iyStart = mesh->yElemCt - 1;
                    diy = -1;
                }

                int ixStart = 0;
                int dix = 1;
                if(quad->mu[ia] < 0)
                {
                    ixStart = mesh->xElemCt - 1;
                    dix = -1;
                }

                int iz = izStart;
                while(iz < (signed) mesh->zElemCt && iz >= 0)
                {
                    int iy = iyStart;
                    while(iy < (signed) mesh->yElemCt && iy >= 0)
                    {
                        int ix = ixStart;
                        while(ix < (signed) mesh->xElemCt && ix >= 0)  // for every mesh element in the proper order
                        {
                            const unsigned int ri = ix*xjmp+iy*yjmp+iz;
                            int zid = mesh->zoneId[ri];  // Get the zone id of this element

                            SOL_T inscatter = 0;
                            for(unsigned int iap = 0; iap < quad->angleCount(); iap++)
                                for(unsigned int il = 0; il <= pn; il++)
                                    inscatter += m_4pi_inv*legendre.table(ia, iap, il) *
                                            xs->scatxs2d(zid, ie, ie, il) *
                                            (*angularFlux)[ie*ejmp + iap*ajmp + ri] *
                                            quad->wt[iap] *
                                            mesh->atomDensity[ri] * mesh->vol[ri];

                            // Handle the x influx
                            if(quad->mu[ia] >= 0)                                       // Approach x = 0 -> xMesh
                            {
                                if(ix == 0)                                               // If this is a boundary cell
                                    influxX = 0.0f;                                       // then the in-flux is zero
                                else                                                      // otherwise
                                    influxX = outboundFluxX[(ix-1)*xjmp + iy*yjmp + iz];  // the in-flux is the out-flux from the previous cell
                            }
                            else                                                          // Approach x = xMesh-1 -> 0
                            {
                                if(ix == (signed) mesh->xElemCt-1)
                                    influxX = 0.0f;
                                else
                                    influxX = outboundFluxX[(ix+1)*xjmp + iy*yjmp + iz];
                            }

                            // Handle the y influx
                            if(quad->zi[ia] >= 0)                                       // Approach y = 0 -> yMesh
                            {
                                if(iy == 0)
                                    influxY = 0.0f;
                                else
                                    influxY = outboundFluxY[ix*xjmp + (iy-1)*yjmp + iz];
                            }
                            else                                                          // Approach y = yMesh-1 -> 0
                            {
                                if(iy == (signed) mesh->yElemCt-1)
                                    influxY = 0.0f;
                                else
                                    influxY = outboundFluxY[ix*xjmp + (iy+1)*yjmp + iz];
                            }

                            // Handle the z influx
                            if(quad->eta[ia] >= 0)
                            {
                                if(iz == 0)
                                    influxZ = 0.0f;
                                else
                                    influxZ = outboundFluxZ[ix*xjmp + iy*yjmp + iz-1];
                            }
                            else
                            {
                                if(iz == (signed) mesh->zElemCt-1)
                                    influxZ = 0.0f;
                                else
                                    influxZ = outboundFluxZ[ix*xjmp + iy*yjmp + iz+1];
                            }

                            if(iterNum == 1 && ri == mesh->voxelCount()/2 && ia==0)
                            {
                                qDebug() << "Zoom";
                                SOL_T v = totalSource[ia*mesh->voxelCount() + ri];

                            }

                            SOL_T numer = totalSource[ia*mesh->voxelCount() + ri] + inscatter +                                                                                             // [#]
                                    mesh->Ayz[ie*quad->angleCount()*mesh->yElemCt*mesh->zElemCt + ia*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz] * influxX +  // [cm^2 * #/cm^2]  The 2x is already factored in
                                    mesh->Axz[ie*quad->angleCount()*mesh->xElemCt*mesh->zElemCt + ia*mesh->xElemCt*mesh->zElemCt + ix*mesh->zElemCt + iz] * influxY +
                                    mesh->Axy[ie*quad->angleCount()*mesh->xElemCt*mesh->yElemCt + ia*mesh->xElemCt*mesh->yElemCt + ix*mesh->yElemCt + iy] * influxZ;
                            SOL_T denom = mesh->vol[ix*xjmp+iy*yjmp+iz]*xs->totXs1d(zid, ie)*mesh->atomDensity[ix*xjmp + iy*yjmp + iz] +                               // [cm^3] * [b] * [1/b-cm]
                                    mesh->Ayz[ie*quad->angleCount()*mesh->yElemCt*mesh->zElemCt + ia*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz] +            // [cm^2]
                                    mesh->Axz[ie*quad->angleCount()*mesh->xElemCt*mesh->zElemCt + ia*mesh->xElemCt*mesh->zElemCt + ix*mesh->zElemCt + iz] +
                                    mesh->Axy[ie*quad->angleCount()*mesh->xElemCt*mesh->yElemCt + ia*mesh->xElemCt*mesh->yElemCt + ix*mesh->yElemCt + iy];

                            //   [#/cm^2] = [#]  / [cm^2]
                            SOL_T cellAvgAngFlux = numer/denom;

                            if(std::isnan(cellAvgAngFlux))
                            {
                                qDebug() << "Found a nan!";
                                qDebug() << "Vol = " << mesh->vol[ix*xjmp+iy*yjmp+iz];
                                qDebug() << "xs = " << xs->totXs1d(zid, ie);
                                qDebug() << "Ayz = " << mesh->Ayz[ia*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz];
                                qDebug() << "Axz = " << mesh->Axz[ia*mesh->xElemCt*mesh->zElemCt + ix*mesh->zElemCt + iz];
                                qDebug() << "Axy = " << mesh->Axy[ia*mesh->xElemCt*mesh->yElemCt + ix*mesh->yElemCt + iy];
                            }

                            SOL_T outx = 2*cellAvgAngFlux - influxX;
                            SOL_T outy = 2*cellAvgAngFlux - influxY;
                            SOL_T outz = 2*cellAvgAngFlux - influxZ;

                            bool cflag = false;
                            if(outx < 0)
                            {
                                cflag = true;
                                outx = 0;
                            }
                            if(outy < 0)
                            {
                                cflag = true;
                                outy = 0;
                            }
                            if(outz < 0)
                            {
                                cflag = true;
                                outz = 0;
                            }
                            if(cflag)
                            {
                                cellAvgAngFlux = (influxX + influxY + influxZ + outx + outy + outz)/6.0;
                            }

                            (*angularFlux)[ie*ejmp + ia*ajmp + ix*xjmp + iy*yjmp + iz] = cellAvgAngFlux;

                            outboundFluxX[ix*xjmp + iy*yjmp + iz] = outx;
                            outboundFluxY[ix*xjmp + iy*yjmp + iz] = outy;
                            outboundFluxZ[ix*xjmp + iy*yjmp + iz] = outz;

                            // Sum all the angular fluxes
                            tempFlux[ix*xjmp + iy*yjmp + iz] += quad->wt[ia]*cellAvgAngFlux;

                            ix += dix;
                        } // end of for ix

                        iy += diy;
                    } // end of for iy

                    iz += diz;
                } // end of for iz

                //float sm = 0.0f;
                //for(unsigned int i = 0; i < tempFlux.size(); i++)
                //    sm += tempFlux[i];

                for(unsigned int i = 0; i < tempFlux.size(); i++)
                {
                    //int indx = ie*m_mesh->voxelCount() + i; // TODO - delete
                    (*scalarFlux)[ie*mesh->voxelCount() + i] = tempFlux[i];
                }
                emit signalNewSolverIteration(scalarFlux);

                unsigned int xTracked = mesh->xElemCt/2;
                unsigned int yTracked = mesh->yElemCt/2;
                unsigned int zTracked = mesh->zElemCt/2;
                converganceTracker.push_back((*scalarFlux)[ie*mesh->voxelCount() + xTracked*xjmp + yTracked*yjmp + zTracked]);

            } // end of all angles

            maxDiff = -1.0E35f;
            for(unsigned int i = 0; i < tempFlux.size(); i++)
            {
                maxDiff = qMax(maxDiff, qAbs((tempFlux[i] - preFlux[i])/tempFlux[i]));

                if(std::isnan(maxDiff))
                    qDebug() << "Found a diff nan!";
            }
            qDebug() << "Max diff = " << maxDiff;

            errList[ie].push_back(maxDiff);
            errMaxList[ie] = maxDiff;
            converganceIters[ie] = iterNum;

            for(unsigned int i = 0; i < tempFlux.size(); i++)
            {
                (*scalarFlux)[ie*mesh->voxelCount() + i] = tempFlux[i];
            }

            iterNum++;
            //emit signalNewIteration(scalarFlux);
        } // end not converged

        //emit signalNewIteration(scalarFlux);
    }  // end each energy group

    qDebug() << "Time to complete: " << (std::clock() - startTime)/(double)(CLOCKS_PER_SEC/1000.0) << " ms";

    //qDebug() << "Convergance of 128, 128, 32:";
    //for(unsigned int i = 0; i < converganceTracker.size(); i++)\
    //{
    //    qDebug() << i << "\t" << converganceTracker[i];
    //}
    //qDebug() << "";

    for(unsigned int i = 0; i < errList.size(); i++)
    {
        qDebug() << "Group: " << i << "   maxDiff: " << errMaxList[i];
        qDebug() << "Iterations: " << converganceIters[i];
        for(unsigned int j = 0; j < errList[i].size(); j++)
            std::cout << errList[i][j] << "\t";
        std::cout << "\n" << std::endl;
    }

    emit signalNewSolverIteration(scalarFlux);
    emit signalSolverFinished(angularFlux);
}

void Solver::raytraceHarmonic(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const unsigned int pn)
{
    std::clock_t startMoment = std::clock();

    unsigned int groups = xs->groupCount();

    RAY_T sx = 25.3906f;
    RAY_T sy = 50.0f - 46.4844f;
    RAY_T sz = 6.8906f;

    unsigned int ejmp = mesh->voxelCount();
    unsigned int xjmp = mesh->xjmp();
    unsigned int yjmp = mesh->yjmp();

    std::vector<RAY_T> *uflux = basicRaytrace(quad, mesh, xs, sx, sy, sz);

    qDebug() << "Time to complete raytracer: " << (std::clock() - startMoment)/(double)(CLOCKS_PER_SEC/1000) << " ms";

    std::vector<RAY_T> *moments = new std::vector<RAY_T>;
    unsigned int momentCount = (pn + 1) * (pn + 1);
    moments->resize(groups * mesh->voxelCount() * momentCount, 0.0f);

    int epjmp = mesh->voxelCount() * momentCount;
    int xpjmp = mesh->yElemCt * mesh->zElemCt * momentCount;
    int ypjmp = mesh->zElemCt * momentCount;
    int zpjmp = momentCount;

    SphericalHarmonic harmonic;

    for(unsigned int ie = 0; ie < groups; ie++)
        for(unsigned int iz = 0; iz < mesh->zElemCt; iz++)
            for(unsigned int iy = 0; iy < mesh->yElemCt; iy++)
                for(unsigned int ix = 0; ix < mesh->xElemCt; ix++)  // For every voxel
                {
                    RAY_T x = mesh->xNodes[ix] + mesh->dx[ix]/2;
                    RAY_T y = mesh->yNodes[iy] + mesh->dy[iy]/2;
                    RAY_T z = mesh->zNodes[iz] + mesh->dz[iz]/2;

                    RAY_T deltaX = x - sx;
                    RAY_T deltaY = y - sy;
                    RAY_T deltaZ = z - sz;

                    // normalize to unit vector
                    RAY_T mag = sqrt(deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ);
                    deltaX /= mag;
                    deltaY /= mag;
                    deltaZ /= mag;

                    RAY_T theta = atan(deltaY / deltaX);
                    RAY_T phi = acos(deltaZ);

                    for(unsigned int il = 0; il <= pn; il++)
                    {
                        //const unsigned int il2 = il * il;
                        (*moments)[ie*epjmp + ix*xpjmp + iy*ypjmp + iz*zpjmp + il*il] = (*uflux)[ie*ejmp + ix*xjmp + iy*yjmp + iz] * harmonic.yl0(il, theta, phi);
                        for(unsigned int im = 1; im <= il; im++)
                        {
                            (*moments)[ie*epjmp + ix*xpjmp + iy*ypjmp + iz*zpjmp + il*il + 2*im - 1] = (*uflux)[ie*ejmp + ix*xjmp + iy*yjmp + iz] * harmonic.ylm_o(il, im, theta, phi);
                            (*moments)[ie*epjmp + ix*xpjmp + iy*ypjmp + iz*zpjmp + il*il + 2*im]     = (*uflux)[ie*ejmp + ix*xjmp + iy*yjmp + iz] * harmonic.ylm_e(il, im, theta, phi);
                        }
                    }
                }

    qDebug() << "Raytracer + moment mapping completed in " << (std::clock() - startMoment)/(double)(CLOCKS_PER_SEC/1000) << " ms";

    emit signalRaytracerFinished(moments);
    emit signalNewRaytracerIteration(uflux);
}


void Solver::gsSolverHarmonic(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const unsigned int pn, const std::vector<RAY_T> *umoments)
{

    // Do some input checks
    if(pn > 10)
    {
        qDebug() << "pn check failed, pn = " << pn;
        return;
    }

    std::clock_t startMoment = std::clock();

    const int maxIterations = 25;
    const SOL_T epsilon = 0.01f;
    const unsigned int momentCount = (pn + 1) * (pn + 1);
    SphericalHarmonic harmonic;

    //std::vector<SOL_T> angularFlux(xs->groupCount() * quad->angleCount() * mesh->voxelCount());
    std::vector<SOL_T> *scalarFlux = new std::vector<SOL_T>(xs->groupCount() * mesh->voxelCount(), 0.0f);
    std::vector<SOL_T> *moments = new std::vector<SOL_T>(xs->groupCount() * mesh->voxelCount() * momentCount, 0.0f);
    std::vector<SOL_T> tempFlux(mesh->voxelCount());
    std::vector<SOL_T> preFlux(mesh->voxelCount(), -100.0f);
    std::vector<SOL_T> totalSource(mesh->voxelCount(), -100.0f);
    std::vector<SOL_T> outboundFluxX(mesh->voxelCount(), -100.0f);
    std::vector<SOL_T> outboundFluxY(mesh->voxelCount(), -100.0f);
    std::vector<SOL_T> outboundFluxZ(mesh->voxelCount(), -100.0f);
    std::vector<SOL_T> extSource(xs->groupCount() * mesh->voxelCount(), 0.0f);
    std::vector<SOL_T> momentToDiscrete(quad->angleCount() * momentCount);

    std::vector<SOL_T> errMaxList;
    std::vector<std::vector<SOL_T> > errList;
    std::vector<int> converganceIters;
    std::vector<SOL_T> converganceTracker;

    errMaxList.resize(xs->groupCount());
    errList.resize(xs->groupCount());
    converganceIters.resize(xs->groupCount());
    converganceTracker.resize(xs->groupCount());

    const XSection &xsref = *xs;

    SOL_T influxX = 0.0f;
    SOL_T influxY = 0.0f;
    SOL_T influxZ = 0.0f;

    int ejmp = mesh->voxelCount() * quad->angleCount();
    int ajmp = mesh->voxelCount();
    int xjmp = mesh->xjmp();
    int yjmp = mesh->yjmp();

    // Compute the moment-to-discrete matrix
    for(unsigned int ia = 0; ia < quad->angleCount(); ia++)
    {
        SOL_T theta = acos(quad->zi[ia]);
        SOL_T phi = atan(quad->eta[ia] / quad->mu[ia]);

        for(unsigned int il = 0; il <= pn; il++)
        {
            momentToDiscrete[ia * momentCount + il*il] = harmonic.yl0(il, theta, phi);
            for(unsigned int im = 1; im <= il; im++)
            {
                momentToDiscrete[ia*momentCount + il*il + 2*im-1] = harmonic.ylm_o(il, im, theta, phi);
                momentToDiscrete[ia*momentCount + il*il + 2*im] = harmonic.ylm_e(il, im, theta, phi);
            }
        }
    }

    // Try to skip over any empty energy groups
    bool noDownscatterYet = true;
    unsigned int highestEnergy = 0;

    while(noDownscatterYet)
    {
        if(highestEnergy >= xs->groupCount())
        {
            qDebug() << "ERROR: No moment data found!";
            break;
        }

        SOL_T dmax = 0.0;
        unsigned int vc = mesh->voxelCount() * momentCount;
        for(unsigned int ira = 0; ira < vc; ira++)
        {
            dmax = (dmax > (*umoments)[highestEnergy*vc + ira]) ? dmax : (*umoments)[highestEnergy*vc + ira];
        }
        if(dmax <= 0.0)
        {
            qDebug() << "No external source or downscatter, skipping energy group " << highestEnergy;
            highestEnergy++;
        }
        else
        {
            noDownscatterYet = false;
        }
    }

    qDebug() << "Solver::gsSolverHarmonic solving " << mesh->voxelCount() * momentCount * xs->groupCount() << " elements in phase space";

    for(unsigned int ie = 0; ie < xs->groupCount(); ie++)  // for every energy group
    {
        /*
        if(!downscatterFlag)
        {
            SOL_T dmax = 0.0;
            int vc = mesh->voxelCount();
            for(unsigned int ri = 0; ri < mesh->voxelCount(); ri++)
            {
                dmax = (dmax > extSource[ie*vc + ri]) ? dmax : extSource[ie*vc + ri];
            }
            if(dmax <= 0.0)
            {
                qDebug() << "No external source or downscatter, skipping energy group " << ie;
                continue;
            }
        }
        downscatterFlag = true;

        qDebug() << "Energy group #" << ie;
        // Do the solving...
        */

        int iterNum = 1;
        SOL_T maxDiff = 1.0;

        while(iterNum <= maxIterations && maxDiff > epsilon)  // while not converged
        {
            qDebug() << "Iteration #" << iterNum;

            preFlux = tempFlux;  // Store flux for previous iteration

            for(unsigned int i = 0; i < totalSource.size(); i++)
                totalSource[i] = 0;

            // Calculate the scattering source
            // TODO
            for(unsigned int iie = 0; iie <= ie; iie++)
                for(int iik = 0; iik < (signed) mesh->zElemCt; iik++)
                    for(int iij = 0; iij < (signed)mesh->yElemCt; iij++)
                        for(int iii = 0; iii < (signed)mesh->xElemCt; iii++)
                        {
                            int indx = iii*xjmp + iij*yjmp + iik;
                            int zidIndx = mesh->zoneId[indx];

                            //         [#]    +=  [#]          *      [#/cm^2]                               * [b]                          * [1/b-cm]                * [cm^3]
                            totalSource[indx] += 1.0/(m_4pi)*(*scalarFlux)[iie*mesh->voxelCount() + indx] * xsref.scatXs1d(zidIndx, iie) * mesh->atomDensity[indx] * mesh->vol[indx]; //xsref(ie-1, zidIndx, 0, iie));
                        }

            // Calculate the total source
            for(unsigned int ri = 0; ri < mesh->voxelCount(); ri++)
            {
                //  [#]         +=  [#]
                totalSource[ri] += extSource[ie*mesh->voxelCount() + ri];
            }

            // Clear for a new sweep
            for(unsigned int i = 0; i < tempFlux.size(); i++)
                tempFlux[i] = 0;

            for(unsigned int iang = 0; iang < quad->angleCount(); iang++)  // for every angle
            {
                qDebug() << "Angle #" << iang;

                // Find the correct direction to sweep
                int izStart = 0;                  // Sweep start index
                int diz = 1;                      // Sweep direction
                if(quad->eta[iang] < 0)           // Condition to sweep backward
                {
                    izStart = mesh->zElemCt - 1;  // Start at the far end
                    diz = -1;                     // Sweep toward zero
                }

                int iyStart = 0;
                int diy = 1;
                if(quad->zi[iang] < 0)
                {
                    iyStart = mesh->yElemCt - 1;
                    diy = -1;
                }

                int ixStart = 0;
                int dix = 1;
                if(quad->mu[iang] < 0)
                {
                    ixStart = mesh->xElemCt - 1;
                    dix = -1;
                }

                int iz = izStart;
                while(iz < (signed) mesh->zElemCt && iz >= 0)
                {
                    int iy = iyStart;
                    while(iy < (signed) mesh->yElemCt && iy >= 0)
                    {
                        int ix = ixStart;
                        while(ix < (signed) mesh->xElemCt && ix >= 0)  // for every mesh element in the proper order
                        {
                            int zid = mesh->zoneId[ix*xjmp + iy*yjmp + iz];  // Get the zone id of this element

                            // Handle the x influx
                            if(quad->mu[iang] >= 0)                                       // Approach x = 0 -> xMesh
                            {
                                if(ix == 0)                                               // If this is a boundary cell
                                    influxX = 0.0f;                                       // then the in-flux is zero
                                else                                                      // otherwise
                                    influxX = outboundFluxX[(ix-1)*xjmp + iy*yjmp + iz];  // the in-flux is the out-flux from the previous cell
                            }
                            else                                                          // Approach x = xMesh-1 -> 0
                            {
                                if(ix == (signed) mesh->xElemCt-1)
                                    influxX = 0.0f;
                                else
                                    influxX = outboundFluxX[(ix+1)*xjmp + iy*yjmp + iz];
                            }

                            // Handle the y influx
                            if(quad->zi[iang] >= 0)                                       // Approach y = 0 -> yMesh
                            {
                                if(iy == 0)
                                    influxY = 0.0f;
                                else
                                    influxY = outboundFluxY[ix*xjmp + (iy-1)*yjmp + iz];
                            }
                            else                                                          // Approach y = yMesh-1 -> 0
                            {
                                if(iy == (signed) mesh->yElemCt-1)
                                    influxY = 0.0f;
                                else
                                    influxY = outboundFluxY[ix*xjmp + (iy+1)*yjmp + iz];
                            }

                            // Handle the z influx
                            if(quad->eta[iang] >= 0)
                            {
                                if(iz == 0)
                                    influxZ = 0.0f;
                                else
                                    influxZ = outboundFluxZ[ix*xjmp + iy*yjmp + iz-1];
                            }
                            else
                            {
                                if(iz == (signed) mesh->zElemCt-1)
                                    influxZ = 0.0f;
                                else
                                    influxZ = outboundFluxZ[ix*xjmp + iy*yjmp + iz+1];
                            }

                            SOL_T numer = totalSource[ix*xjmp+iy*yjmp+iz] +                                                                                              // [#]
                                    mesh->Ayz[ie*quad->angleCount()*mesh->yElemCt*mesh->zElemCt + iang*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz] * influxX +  // [cm^2 * #/cm^2]  The 2x is already factored in
                                    mesh->Axz[ie*quad->angleCount()*mesh->xElemCt*mesh->zElemCt + iang*mesh->xElemCt*mesh->zElemCt + ix*mesh->zElemCt + iz] * influxY +
                                    mesh->Axy[ie*quad->angleCount()*mesh->xElemCt*mesh->yElemCt + iang*mesh->xElemCt*mesh->yElemCt + ix*mesh->yElemCt + iy] * influxZ;
                            SOL_T denom = mesh->vol[ix*xjmp+iy*yjmp+iz]*xsref.totXs1d(zid, ie)*mesh->atomDensity[ix*xjmp + iy*yjmp + iz] +                               // [cm^3] * [b] * [1/b-cm]
                                    mesh->Ayz[ie*quad->angleCount()*mesh->yElemCt*mesh->zElemCt + iang*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz] +            // [cm^2]
                                    mesh->Axz[ie*quad->angleCount()*mesh->xElemCt*mesh->zElemCt + iang*mesh->xElemCt*mesh->zElemCt + ix*mesh->zElemCt + iz] +
                                    mesh->Axy[ie*quad->angleCount()*mesh->xElemCt*mesh->yElemCt + iang*mesh->xElemCt*mesh->yElemCt + ix*mesh->yElemCt + iy];

                            //   [#/cm^2] = [#]  / [cm^2]
                            SOL_T angFlux = numer/denom;

                            std::vector<float> gxs;
                            for(unsigned int i = 0; i < xsref.groupCount(); i++)
                            {
                                gxs.push_back(xsref.totXs1d(zid, i));
                            }

                            if(std::isnan(angFlux))
                            {
                                qDebug() << "Found a nan!";
                                qDebug() << "Vol = " << mesh->vol[ix*xjmp+iy*yjmp+iz];
                                qDebug() << "xs = " << xsref.totXs1d(zid, ie);
                                qDebug() << "Ayz = " << mesh->Ayz[iang*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz];
                                qDebug() << "Axz = " << mesh->Axz[iang*mesh->xElemCt*mesh->zElemCt + ix*mesh->zElemCt + iz];
                                qDebug() << "Axy = " << mesh->Axy[iang*mesh->xElemCt*mesh->yElemCt + ix*mesh->yElemCt + iy];
                            }

                            //angularFlux[ie*ejmp + iang*ajmp + ix*xjmp + iy*yjmp + iz] = angFlux;

                            outboundFluxX[ix*xjmp + iy*yjmp + iz] = 2*angFlux - influxX;
                            outboundFluxY[ix*xjmp + iy*yjmp + iz] = 2*angFlux - influxY;
                            outboundFluxZ[ix*xjmp + iy*yjmp + iz] = 2*angFlux - influxZ;

                            // Sum all the angular fluxes
                            tempFlux[ix*xjmp + iy*yjmp + iz] += quad->wt[iang]*angFlux;

                            ix += dix;
                        } // end of for ix

                        iy += diy;
                    } // end of for iy

                    iz += diz;
                } // end of for iz

                //float sm = 0.0f;
                //for(unsigned int i = 0; i < tempFlux.size(); i++)
                //    sm += tempFlux[i];

                for(unsigned int i = 0; i < tempFlux.size(); i++)
                {
                    //int indx = ie*m_mesh->voxelCount() + i; // TODO - delete
                    (*scalarFlux)[ie*mesh->voxelCount() + i] = tempFlux[i];
                }
                emit signalNewSolverIteration(scalarFlux);

                unsigned int xTracked = mesh->xElemCt/2;
                unsigned int yTracked = mesh->yElemCt/2;
                unsigned int zTracked = mesh->zElemCt/2;
                converganceTracker.push_back((*scalarFlux)[ie*mesh->voxelCount() + xTracked*xjmp + yTracked*yjmp + zTracked]);

            } // end of all angles

            maxDiff = -1.0E35f;
            for(unsigned int i = 0; i < tempFlux.size(); i++)
            {
                //float z = qAbs((tempFlux[i] - preFlux[i])/tempFlux[i]);
                maxDiff = qMax(maxDiff, qAbs((tempFlux[i] - preFlux[i])/tempFlux[i]));

                if(std::isnan(maxDiff))
                    qDebug() << "Found a diff nan!";
            }
            qDebug() << "Max diff = " << maxDiff;

            errList[ie].push_back(maxDiff);
            errMaxList[ie] = maxDiff;
            converganceIters[ie] = iterNum;

            for(unsigned int i = 0; i < tempFlux.size(); i++)
            {
                (*scalarFlux)[ie*mesh->voxelCount() + i] = tempFlux[i];
            }

            iterNum++;
            //emit signalNewIteration(scalarFlux);
        } // end not converged

        //emit signalNewIteration(scalarFlux);
    }  // end each energy group

    qDebug() << "Time to complete: " << (std::clock() - startMoment)/(double)(CLOCKS_PER_SEC/1000) << " ms";

    qDebug() << "Convergance of 128, 128, 32:";
    for(unsigned int i = 0; i < converganceTracker.size(); i++)\
    {
        qDebug() << i << "\t" << converganceTracker[i];
    }
    qDebug() << "";

    for(unsigned int i = 0; i < errList.size(); i++)
    {
        qDebug() << "Group: " << i << "   maxDiff: " << errMaxList[i];
        qDebug() << "Iterations: " << converganceIters[i];
        for(unsigned int j = 0; j < errList[i].size(); j++)
            std::cout << errList[i][j] << "\t";
        std::cout << "\n" << std::endl;
    }

    emit signalNewSolverIteration(scalarFlux);
    emit signalSolverFinished(scalarFlux);
}
