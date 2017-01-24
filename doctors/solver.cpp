#define _USE_MATH_DEFINES
#include <cmath>

#include "solver.h"

#include <ctime>

#include <QDebug>
#include <iostream>
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

void Solver::raytraceIso(const Quadrature *quad, const Mesh *mesh, const XSection *xs)
{
    std::clock_t startMoment = std::clock();

    int groups = xs->groupCount();

    const unsigned short DIRECTION_X = 1;
    const unsigned short DIRECTION_Y = 2;
    const unsigned short DIRECTION_Z = 3;

    std::vector<float> *uflux = new std::vector<float>;
    uflux->resize(groups * mesh->voxelCount());

    unsigned int ejmp = mesh->voxelCount();
    unsigned int xjmp = mesh->xjmp();
    unsigned int yjmp = mesh->yjmp();

    qDebug() << "Running raytracer";

    float tiny = 1.0E-35f;
    float huge = 1.0E35f;
    std::vector<float> meanFreePaths;
    meanFreePaths.resize(xs->groupCount());

    float sx = 25.3906f;
    float sy = 50.0f - 46.4844f;
    float sz = 6.8906f;
    //                                0  1  2  3  4  5  6  7  8  9  0  1  2  3  4  5  6  7  8
    std::vector<float> srcStrength = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};

    unsigned int srcIndxX = int(sx / mesh->dx[0]);  // TODO: This will only work for evenly spaced grids
    unsigned int srcIndxY = int(sy / mesh->dy[0]);
    unsigned int srcIndxZ = int(sz / mesh->dz[0]);

    for(unsigned int zIndxStart = 0; zIndxStart < mesh->zElemCt; zIndxStart++)
        for(unsigned int yIndxStart = 0; yIndxStart < mesh->yElemCt; yIndxStart++)
            for(unsigned int xIndxStart = 0; xIndxStart < mesh->xElemCt; xIndxStart++)  // For every voxel
            {
                float x = mesh->xNodes[xIndxStart] + mesh->dx[xIndxStart]/2;
                float y = mesh->yNodes[yIndxStart] + mesh->dy[yIndxStart]/2;
                float z = mesh->zNodes[zIndxStart] + mesh->dz[zIndxStart]/2;

                std::vector<float> tmpdistv;
                std::vector<float> tmpxsv;
                std::vector<float> mfpv;

                if(xIndxStart == srcIndxX && yIndxStart == srcIndxY && zIndxStart == srcIndxZ)  // End condition
                {
                    float srcToCellDist = sqrt((x-sx)*(x-sx) + (y-sy)*(y-sy) + (z-sz)*(z-sz));
                    unsigned int zid = mesh->zoneId[xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart];
                    float xsval;
                    for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                    {
                        xsval = xs->m_tot1d[zid*groups + ie] * mesh->atomDensity[xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart];
                        (*uflux)[ie*ejmp + xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart] = srcStrength[ie] * exp(-xsval*srcToCellDist) / (4 * M_PI * srcToCellDist * srcToCellDist);
                    }
                    continue;
                }

                // Start raytracing through the geometry
                unsigned int xIndx = xIndxStart;
                unsigned int yIndx = yIndxStart;
                unsigned int zIndx = zIndxStart;

                float srcToCellX = sx - x;
                float srcToCellY = sy - y;
                float srcToCellZ = sz - z;

                float srcToCellDist = sqrt(srcToCellX*srcToCellX + srcToCellY*srcToCellY + srcToCellZ*srcToCellZ);

                float xcos = srcToCellX/srcToCellDist;  // Fraction of direction biased in x-direction, unitless
                float ycos = srcToCellY/srcToCellDist;
                float zcos = srcToCellZ/srcToCellDist;

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
                    float tx = (fabs(xcos) < tiny ? huge : (mesh->xNodes[xBoundIndx] - x)/xcos);  // Distance traveled [cm] when next cell is
                    float ty = (fabs(ycos) < tiny ? huge : (mesh->yNodes[yBoundIndx] - y)/ycos);  //   entered traveling in x direction
                    float tz = (fabs(zcos) < tiny ? huge : (mesh->zNodes[zBoundIndx] - z)/zcos);

                    // Determine the shortest distance traveled [cm] before _any_ surface is crossed
                    float tmin;
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
                    tmpdistv.push_back(tmin);
                    tmpxsv.push_back(xs->m_tot1d[zid*groups + 18] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx]);
                    float gain = tmin * xs->m_tot1d[zid*groups + 18] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx];
                    mfpv.push_back(gain);

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
                        y = mesh->yNodes[yBoundIndx];
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
                        z = mesh->zNodes[zBoundIndx];
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

                    if(xIndx == srcIndxX && yIndx == srcIndxY && zIndx == srcIndxZ)
                    {
                        float finalDist = sqrt((x-sx)*(x-sx) + (y-sy)*(y-sy) + (z-sz)*(z-sz));

                        for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                        {
                            //       [#]       = [cm] * [b] * [1/cm-b]
                            meanFreePaths[ie] += finalDist * xs->m_tot1d[zid*groups + ie] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx];
                        }

                        tmpdistv.push_back(finalDist);
                        tmpxsv.push_back(xs->m_tot1d[zid*groups + 18] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx]);

                        float gain = finalDist * xs->m_tot1d[zid*groups + 18] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx];
                        mfpv.push_back(gain);

                        exhaustedRay = true;
                    }

                } // End of while loop

                for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                {
                    float flx = srcStrength[ie] * exp(-meanFreePaths[ie]) / (4 * M_PI * srcToCellDist * srcToCellDist);

                    if(flx < 0)
                        qDebug() << "solver.cpp: (291): Negative?";

                    if(flx > 1E6)
                        qDebug() << "solver.cpp: (294): Too big!";

                    (*uflux)[ie*ejmp + xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart] = flx;  //srcStrength * exp(-meanFreePaths[ie]) / (4 * M_PI * srcToCellDist * srcToCellDist);
                }

            } // End of each voxel

    qDebug() << "Time to complete raytracer: " << (std::clock() - startMoment)/(double)(CLOCKS_PER_SEC/1000) << " ms";

    emit signalRaytracerFinished(uflux);
    emit signalNewIteration(uflux);
}


void Solver::gsSolverIso(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const std::vector<float> *uflux)
{

    std::clock_t startMoment = std::clock();

    const int maxIterations = 25;
    const float epsilon = 0.01f;


    std::vector<float> angularFlux(xs->groupCount() * quad->angleCount() * mesh->voxelCount());
    std::vector<float> *scalarFlux = new std::vector<float>(xs->groupCount() * mesh->voxelCount(), 0.0f);
    std::vector<float> tempFlux(mesh->voxelCount());
    std::vector<float> preFlux(mesh->voxelCount(), -100.0f);
    std::vector<float> totalSource(mesh->voxelCount(), -100.0f);
    std::vector<float> outboundFluxX(mesh->voxelCount(), -100.0f);
    std::vector<float> outboundFluxY(mesh->voxelCount(), -100.0f);
    std::vector<float> outboundFluxZ(mesh->voxelCount(), -100.0f);
    std::vector<float> extSource(xs->groupCount() * mesh->voxelCount(), 0.0f);

    std::vector<float> errMaxList;
    std::vector<std::vector<float> > errList;
    std::vector<int> converganceIters;
    std::vector<float> converganceTracker;

    errMaxList.resize(xs->groupCount());
    errList.resize(xs->groupCount());
    converganceIters.resize(xs->groupCount());
    converganceTracker.resize(xs->groupCount());

    const XSection &xsref = *xs;

    float influxX = 0.0f;
    float influxY = 0.0f;
    float influxZ = 0.0f;

    int ejmp = mesh->voxelCount() * quad->angleCount();
    int ajmp = mesh->voxelCount();
    int xjmp = mesh->xjmp();
    int yjmp = mesh->yjmp();

    bool downscatterFlag = false;

    if(uflux != NULL)
    {
        qDebug() << "Loading uncollided flux into external source";
        // If there is an uncollided flux provided, use it, otherwise, calculate the external source
        for(unsigned int ei = 0; ei < xs->groupCount(); ei++)
            for(unsigned int ri = 0; ri < mesh->voxelCount(); ri++)
                //                              [#]   =                        [#/cm^2]      * [cm^3]        *  [b]                               * [1/b-cm]
                extSource[ei*mesh->voxelCount() + ri] = (*uflux)[ei*mesh->voxelCount() + ri] * mesh->vol[ri] * xs->scatXs1d(mesh->zoneId[ri], ei) * mesh->atomDensity[ri];

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

    qDebug() << "Solver::gssolver(): 379: Solving " << mesh->voxelCount() * quad->angleCount() * xs->groupCount() << " elements in phase space";

    for(unsigned int ie = 0; ie < xs->groupCount(); ie++)  // for every energy group
    {
        if(!downscatterFlag)
        {
            float dmax = 0.0;
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

        int iterNum = 1;
        float maxDiff = 1.0;

        while(iterNum <= maxIterations && maxDiff > epsilon)  // while not converged
        {
            qDebug() << "Iteration #" << iterNum;

            preFlux = tempFlux;  // Store flux for previous iteration

            for(unsigned int i = 0; i < totalSource.size(); i++)
                totalSource[i] = 0;

            // Calculate the scattering source
            for(unsigned int iie = 0; iie <= ie; iie++)
                for(int iik = 0; iik < (signed) mesh->zElemCt; iik++)
                    for(int iij = 0; iij < (signed)mesh->yElemCt; iij++)
                        for(int iii = 0; iii < (signed)mesh->xElemCt; iii++)
                        {
                            int indx = iii*xjmp + iij*yjmp + iik;
                            int zidIndx = mesh->zoneId[indx];

                            //         [#]    +=  [#]          *      [#/cm^2]                               * [b]                          * [1/b-cm]                * [cm^3]
                            totalSource[indx] += 1.0/(4.0*M_PI)*(*scalarFlux)[iie*mesh->voxelCount() + indx] * xsref.scatXs1d(zidIndx, iie) * mesh->atomDensity[indx] * mesh->vol[indx]; //xsref(ie-1, zidIndx, 0, iie));
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

            for(int iang = 0; iang < quad->angleCount(); iang++)  // for every angle
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

                            float numer = totalSource[ix*xjmp+iy*yjmp+iz] +                                                                                              // [#]
                                    mesh->Ayz[ie*quad->angleCount()*mesh->yElemCt*mesh->zElemCt + iang*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz] * influxX +  // [cm^2 * #/cm^2]  The 2x is already factored in
                                    mesh->Axz[ie*quad->angleCount()*mesh->xElemCt*mesh->zElemCt + iang*mesh->xElemCt*mesh->zElemCt + ix*mesh->zElemCt + iz] * influxY +
                                    mesh->Axy[ie*quad->angleCount()*mesh->xElemCt*mesh->yElemCt + iang*mesh->xElemCt*mesh->yElemCt + ix*mesh->yElemCt + iy] * influxZ;
                            float denom = mesh->vol[ix*xjmp+iy*yjmp+iz]*xsref.totXs1d(zid, ie)*mesh->atomDensity[ix*xjmp + iy*yjmp + iz] +                               // [cm^3] * [b] * [1/b-cm]
                                    mesh->Ayz[ie*quad->angleCount()*mesh->yElemCt*mesh->zElemCt + iang*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz] +            // [cm^2]
                                    mesh->Axz[ie*quad->angleCount()*mesh->xElemCt*mesh->zElemCt + iang*mesh->xElemCt*mesh->zElemCt + ix*mesh->zElemCt + iz] +
                                    mesh->Axy[ie*quad->angleCount()*mesh->xElemCt*mesh->yElemCt + iang*mesh->xElemCt*mesh->yElemCt + ix*mesh->yElemCt + iy];

                            //   [#/cm^2] = [#]  / [cm^2]
                            float angFlux = numer/denom;

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

                            angularFlux[ie*ejmp + iang*ajmp + ix*xjmp + iy*yjmp + iz] = angFlux;

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

                float sm = 0.0f;
                for(unsigned int i = 0; i < tempFlux.size(); i++)
                    sm += tempFlux[i];

                for(unsigned int i = 0; i < tempFlux.size(); i++)
                {
                    //int indx = ie*m_mesh->voxelCount() + i; // TODO - delete
                    (*scalarFlux)[ie*mesh->voxelCount() + i] = tempFlux[i];
                }
                emit signalNewIteration(scalarFlux);

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

    emit signalNewIteration(scalarFlux);
    emit signalSolverFinished(scalarFlux);
}


// ////////////////////////////////////////////////////////////////////////////////////////////// //
//                           Anisotropic versions of the above solvers                            //
// ////////////////////////////////////////////////////////////////////////////////////////////// //

void Solver::raytrace(const Quadrature *quad, const Mesh *mesh, const XSection *xs)
{
    std::clock_t startMoment = std::clock();

    unsigned int groups = xs->groupCount();

    const unsigned short DIRECTION_X = 1;
    const unsigned short DIRECTION_Y = 2;
    const unsigned short DIRECTION_Z = 3;

    std::vector<float> *uflux = new std::vector<float>;
    uflux->resize(groups * mesh->voxelCount());

    unsigned int ejmp = mesh->voxelCount();
    unsigned int xjmp = mesh->xjmp();
    unsigned int yjmp = mesh->yjmp();

    qDebug() << "Running raytracer";

    float tiny = 1.0E-35f;
    float huge = 1.0E35f;
    std::vector<float> meanFreePaths;
    meanFreePaths.resize(xs->groupCount());

    float sx = 25.3906f;
    float sy = 50.0f - 46.4844f;
    float sz = 6.8906f;
    //                                0  1  2  3  4  5  6  7  8  9  0  1  2  3  4  5  6  7  8
    std::vector<float> srcStrength = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};

    unsigned int srcIndxX = int(sx / mesh->dx[0]);  // TODO: This will only work for evenly spaced grids
    unsigned int srcIndxY = int(sy / mesh->dy[0]);
    unsigned int srcIndxZ = int(sz / mesh->dz[0]);

    for(unsigned int zIndxStart = 0; zIndxStart < mesh->zElemCt; zIndxStart++)
        for(unsigned int yIndxStart = 0; yIndxStart < mesh->yElemCt; yIndxStart++)
            for(unsigned int xIndxStart = 0; xIndxStart < mesh->xElemCt; xIndxStart++)  // For every voxel
            {
                float x = mesh->xNodes[xIndxStart] + mesh->dx[xIndxStart]/2;
                float y = mesh->yNodes[yIndxStart] + mesh->dy[yIndxStart]/2;
                float z = mesh->zNodes[zIndxStart] + mesh->dz[zIndxStart]/2;

                //float deltaX = x - srcIndxX;
                //float deltaY = y - srcIndxY;
                //float deltaZ = z - srcIndxZ;

                //std::vector<float> tmpdistv;
                //std::vector<float> tmpxsv;
                //std::vector<float> mfpv;

                if(xIndxStart == srcIndxX && yIndxStart == srcIndxY && zIndxStart == srcIndxZ)  // End condition
                {
                    float srcToCellDist = sqrt((x-sx)*(x-sx) + (y-sy)*(y-sy) + (z-sz)*(z-sz));
                    unsigned int zid = mesh->zoneId[xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart];
                    float xsval;
                    for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                    {
                        xsval = xs->m_tot1d[zid*groups + ie] * mesh->atomDensity[xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart];
                        (*uflux)[ie*ejmp + xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart] = srcStrength[ie] * exp(-xsval*srcToCellDist) / (4 * M_PI * srcToCellDist * srcToCellDist);
                        //float magnitude = srcStrength[ie] * exp(-xsval*srcToCellDist) / (4 * M_PI * srcToCellDist * srcToCellDist);
                        //for(unsigned int il = 0; il < lSize; il++)
                        //{
                        //    for(unsigned int im = -il; im < il; im++)
                        //    (*uflux)[ie*ejmp + xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart*momentBins + im] = magnitude * harmonics(l, m, theta, phi);
                        //}
                    }
                    continue;
                }

                // Start raytracing through the geometry
                unsigned int xIndx = xIndxStart;
                unsigned int yIndx = yIndxStart;
                unsigned int zIndx = zIndxStart;

                float srcToCellX = sx - x;
                float srcToCellY = sy - y;
                float srcToCellZ = sz - z;

                float srcToCellDist = sqrt(srcToCellX*srcToCellX + srcToCellY*srcToCellY + srcToCellZ*srcToCellZ);

                float xcos = srcToCellX/srcToCellDist;  // Fraction of direction biased in x-direction, unitless
                float ycos = srcToCellY/srcToCellDist;
                float zcos = srcToCellZ/srcToCellDist;

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
                    float tx = (fabs(xcos) < tiny ? huge : (mesh->xNodes[xBoundIndx] - x)/xcos);  // Distance traveled [cm] when next cell is
                    float ty = (fabs(ycos) < tiny ? huge : (mesh->yNodes[yBoundIndx] - y)/ycos);  //   entered traveling in x direction
                    float tz = (fabs(zcos) < tiny ? huge : (mesh->zNodes[zBoundIndx] - z)/zcos);

                    // Determine the shortest distance traveled [cm] before _any_ surface is crossed
                    float tmin;
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
                    //float gain = tmin * xs->m_tot1d[zid*groups + 18] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx];
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
                        y = mesh->yNodes[yBoundIndx];
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
                        z = mesh->zNodes[zBoundIndx];
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

                    if(xIndx == srcIndxX && yIndx == srcIndxY && zIndx == srcIndxZ)
                    {
                        float finalDist = sqrt((x-sx)*(x-sx) + (y-sy)*(y-sy) + (z-sz)*(z-sz));

                        for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                        {
                            //       [#]       = [cm] * [b] * [1/cm-b]
                            meanFreePaths[ie] += finalDist * xs->m_tot1d[zid*groups + ie] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx];
                        }

                        //tmpdistv.push_back(finalDist);
                        //tmpxsv.push_back(xs->m_tot1d[zid*groups + 18] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx]);

                        //float gain = finalDist * xs->m_tot1d[zid*groups + 18] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx];
                        //mfpv.push_back(gain);

                        exhaustedRay = true;
                    }

                } // End of while loop

                for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                {
                    float flx = srcStrength[ie] * exp(-meanFreePaths[ie]) / (4 * M_PI * srcToCellDist * srcToCellDist);

                    if(flx < 0)
                        qDebug() << "solver.cpp: (291): Negative?";

                    if(flx > 1E6)
                        qDebug() << "solver.cpp: (294): Too big!";

                    (*uflux)[ie*ejmp + xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart] = flx;  //srcStrength * exp(-meanFreePaths[ie]) / (4 * M_PI * srcToCellDist * srcToCellDist);
                }

            } // End of each voxel

    qDebug() << "Time to complete raytracer: " << (std::clock() - startMoment)/(double)(CLOCKS_PER_SEC/1000) << " ms";

    //const int lSize = pn + 1;
    //const int mSize = 2* pn + 1;
    //const int momentBins = lSize * mSize;
    //SphericalHarmonic harmonics;

    std::vector<float> *ufluxAniso = new std::vector<float>;
    ufluxAniso->resize(groups * quad->angleCount() * mesh->voxelCount(), 0.0f);

    int eajmp = quad->angleCount() * mesh->voxelCount();
    //int aajmp = mesh->voxelCount();
    int xajmp = mesh->yNodeCt * mesh->zNodeCt;
    int yajmp = mesh->zNodeCt;

    for(unsigned int ie = 0; ie < groups; ie++)
        for(unsigned int iz = 0; iz < mesh->zElemCt; iz++)
            for(unsigned int iy = 0; iy < mesh->yElemCt; iy++)
                for(unsigned int ix = 0; ix < mesh->xElemCt; ix++)  // For every voxel
                {
                    float x = mesh->xNodes[ix] + mesh->dx[ix]/2;
                    float y = mesh->yNodes[iy] + mesh->dy[iy]/2;
                    float z = mesh->zNodes[iz] + mesh->dz[iz]/2;

                    float deltaX = x - srcIndxX;
                    float deltaY = y - srcIndxY;
                    float deltaZ = z - srcIndxZ;

                    // normalize to unit vector
                    float mag = sqrt(deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ);
                    deltaX /= mag;
                    deltaY /= mag;
                    deltaZ /= mag;

                    //float phi = acos(deltaZ);  // In radians
                    //float theta = acos(x/sin(phi));

                    int indx = ie*eajmp + ix*xajmp + iy*yajmp + iz;  // index of start of moments
                    //for(int il = 0; il < lSize; il++)

                    unsigned int bestAngIndx = 0;
                    float bestCosT = 0.0f;

                    for(unsigned int ia = 0; ia < quad->angleCount(); ia++)
                    {
                        float cosT = quad->mu[ia] * deltaX + quad->eta[ia] * deltaY + quad->zi[ia] * deltaZ;
                        if(cosT > bestCosT)
                        {
                            bestCosT = cosT;
                            bestAngIndx = ia;
                        }
                    }

                    // The rest of the directions remain zero
                    (*ufluxAniso)[indx + ia*mesh->voxelCount()] = (*uflux)[ie*ejmp + ix*xjmp + iy*yjmp + iz];

                    //for(unsigned int ia = 0; ia < quad->angleCount(); ia++)
                    //{

                        //int ildx = il * il; // Offset for il index

                        // P_il^0


                        /*
                        for(int im = 1; im <= il; im++)
                        {
                            int imdx = 2*im - 1;  // Offset for im index

                            if(indx + ildx + imdx + 1 >= ufluxAniso->size())
                            {
                                qDebug() << "Failed indexing check!";
                            }

                            (*ufluxAniso)[indx + ildx + imdx] = (*uflux)[ie*ejmp + ix*xjmp + iy*yjmp + iz] * harmonics.ylm_o(il, im, theta, phi);
                            (*ufluxAniso)[indx + ildx + imdx + 1] = (*uflux)[ie*ejmp + ix*xjmp + iy*yjmp + iz] * harmonics.ylm_e(il, im, theta, phi);
                        }
                        */
                    //}
                }



    emit signalRaytracerFinished(ufluxAniso);
    emit signalNewIteration(uflux);
}


void Solver::gssolver(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const unsigned int pn, const std::vector<float> *uFlux)
{

    // Do some input checks
    if(pn > 10)
    {
        qDebug() << "pn check failed, pn = " << pn;
        return;
    }

    std::clock_t startMoment = std::clock();

    const int maxIterations = 25;
    const float epsilon = 0.01f;
    //const unsigned int momentCount = (pn+1) * (pn+1);
    //SphericalHarmonic harmonic;


    std::vector<float> angularFlux(xs->groupCount() * quad->angleCount() * mesh->voxelCount());
    //std::vector<float> moments(xs->groupCount() * mesh->voxelCount() * momentCount);
    std::vector<float> *scalarFlux = new std::vector<float>(xs->groupCount() * mesh->voxelCount(), 0.0f);
    std::vector<float> tempFlux(mesh->voxelCount());
    std::vector<float> preFlux(mesh->voxelCount(), -100.0f);
    std::vector<float> totalSource(mesh->voxelCount(), -100.0f);
    std::vector<float> outboundFluxX(mesh->voxelCount(), -100.0f);
    std::vector<float> outboundFluxY(mesh->voxelCount(), -100.0f);
    std::vector<float> outboundFluxZ(mesh->voxelCount(), -100.0f);
    std::vector<float> extSource(xs->groupCount() * quad->angleCount() * mesh->voxelCount(), 0.0f);

    std::vector<float> errMaxList;
    std::vector<std::vector<float> > errList;
    std::vector<int> converganceIters;
    std::vector<float> converganceTracker;

    errMaxList.resize(xs->groupCount());
    errList.resize(xs->groupCount());
    converganceIters.resize(xs->groupCount());
    converganceTracker.resize(xs->groupCount());

    const XSection &xsref = *xs;

    float influxX = 0.0f;
    float influxY = 0.0f;
    float influxZ = 0.0f;

    int ejmp = mesh->voxelCount() * quad->angleCount();
    int ajmp = mesh->voxelCount();
    int xjmp = mesh->xjmp();
    int yjmp = mesh->yjmp();

    bool downscatterFlag = false;

    if(uFlux != NULL)
    {
        qDebug() << "Computing 1st collision source";
        // If there is an uncollided flux provided, use it, otherwise, calculate the external source
        for(unsigned int ei = 0; ei < xs->groupCount(); ei++)
            for(unsigned int ai = 0; ai < quad->angleCount(); ai++)
                for(unsigned int ri = 0; ri < mesh->voxelCount(); ri++)
                {
                    float firstColSrc = 0.0f;
                    // TODO - should the equality condition be there?
                    for(unsigned int epi = 0; epi <= ei; epi++)  // For every higher energy that can downscatter
                        for(unsigned int l = 0; l < pn; l++)  // For every Legendre expansion coeff
                        {
                            float legendre_coeff = (2*l + 1) / M_4PI * xs->scatxs2d(mesh->zoneId[ri], epi, ei, l);  // [b]
                            float integral = 0.0f;
                            for(unsigned int api = 0; api < quad->angleCount(); api++) // For every angle
                                integral += legendre(ai, api, l) * uFlux[ei*ejmp + ai*ajmp + ri] * quad->wt[api];
                            // [b/cm^2]  = [b]  * [1/cm^2]
                            firstColSrc += legendre_coeff * integral;
                        }


                    //for(unsigned int li = 0; li < momentCount; li++)
                    //                          [#]   =    [b/cm^2]      * [cm^3]          * [1/b-cm]
                    extSource[ei*ejmp + ai*ajmp + ri] = firstColSrc * mesh->vol[ri] * mesh->atomDensity[ri];  //(*uFlux)[ei*ejmp + ai*ajmp + ri] * mesh->vol[ri] * xs->scatXs1d(mesh->zoneId[ri], ei) * mesh->atomDensity[ri];
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

    for(unsigned int ie = 0; ie < xs->groupCount(); ie++)  // for every energy group
    {
        if(!downscatterFlag)
        {
            float dmax = 0.0;
            unsigned int vc = mesh->voxelCount() * quad->angleCount();
            for(unsigned int ri = 0; ri < vc; ri++)
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

        int iterNum = 1;
        float maxDiff = 1.0;

        while(iterNum <= maxIterations && maxDiff > epsilon)  // while not converged
        {
            qDebug() << "Iteration #" << iterNum;

            preFlux = tempFlux;  // Store flux for previous iteration

            for(unsigned int i = 0; i < totalSource.size(); i++)
                totalSource[i] = 0;

            // Calculate the scattering source
            /*
            for(unsigned int iie = 0; iie <= ie; iie++)
                for(int iik = 0; iik < (signed) mesh->zElemCt; iik++)
                    for(int iij = 0; iij < (signed)mesh->yElemCt; iij++)
                        for(int iii = 0; iii < (signed)mesh->xElemCt; iii++)
                        {
                            int indx = iii*xjmp + iij*yjmp + iik;
                            int zidIndx = mesh->zoneId[indx];

                            //         [#]    +=  [#]          *      [#/cm^2]                               * [b]                          * [1/b-cm]                * [cm^3]
                            totalSource[indx] += 1.0/(4.0*M_PI)*(*scalarFlux)[iie*mesh->voxelCount() + indx] * xsref.scatXs1d(zidIndx, iie) * mesh->atomDensity[indx] * mesh->vol[indx]; //xsref(ie-1, zidIndx, 0, iie));
                        }

            // Calculate the total source
            for(unsigned int ri = 0; ri < mesh->voxelCount(); ri++)
            {
                //  [#]         +=  [#]
                totalSource[ri] += extSource[ie*mesh->voxelCount() + ri];
            }
            */

            // Clear for a new sweep
            for(unsigned int i = 0; i < tempFlux.size(); i++)
                tempFlux[i] = 0;

            for(int iang = 0; iang < quad->angleCount(); iang++)  // for every angle
            {
                qDebug() << "Angle #" << iang;
                float theta = acos(quad->zi[iang]);
                float phi = acos(quad->mu[iang]/sqrt(1 - quad->zi[iang]*quad->zi[iang]));

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

                            // Compute the source
                            float src_era = 0.0f;  // source for this energy, voxel, angle
                            for(unsigned int iie = 0; iie <= ie; iie++)  // For every group higher
                             {
                                for(unsigned int il = 0; il <= pn; il++)  // For every Legendre projection
                                {
                                    float sig_slgg = xs->scatxs2d(zid, iie, ie, il);
                                    float integral = 0.0f;

                                    float moment_l0e = 0.0f;

                                    // TODO compute moment l0e

                                    integral += harmonic.ylm_e(il, 0, theta, phi) * moment_l0e;

                                    for(unsigned int im = 1; im < il; im++)
                                    {
                                        float moment_lme = 0.0f;
                                        float moment_lmo = 0.0f;

                                        // TODO compute moments

                                        integral += harmonic.ylm_e(il, im, theta, phi) * moment_lme + harmonic.ylm_o(il, im, theta, phi) * moment_lmo;
                                    }

                                    src_era += sig_slgg * integral;

                                    //ei*mesh->voxelCount()*momentCount + ri*momentCount + li
                                    //         [#]    +=  [#]          *      [#/cm^2]                               * [b]                          * [1/b-cm]                * [cm^3]
                                    //src_era = extSource[];
                                    //totalSource[1] += 1.0/(4.0*M_PI)*(*scalarFlux)[iie*mesh->voxelCount() + 1] * xsref.scatXs1d(zid, iie) * mesh->atomDensity[1] * mesh->vol[1]; //xsref(ie-1, zidIndx, 0, iie));
                                }
                            }

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

                            float numer = totalSource[ix*xjmp+iy*yjmp+iz] +                                                                                              // [#]
                                    mesh->Ayz[ie*quad->angleCount()*mesh->yElemCt*mesh->zElemCt + iang*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz] * influxX +  // [cm^2 * #/cm^2]  The 2x is already factored in
                                    mesh->Axz[ie*quad->angleCount()*mesh->xElemCt*mesh->zElemCt + iang*mesh->xElemCt*mesh->zElemCt + ix*mesh->zElemCt + iz] * influxY +
                                    mesh->Axy[ie*quad->angleCount()*mesh->xElemCt*mesh->yElemCt + iang*mesh->xElemCt*mesh->yElemCt + ix*mesh->yElemCt + iy] * influxZ;
                            float denom = mesh->vol[ix*xjmp+iy*yjmp+iz]*xsref.totXs1d(zid, ie)*mesh->atomDensity[ix*xjmp + iy*yjmp + iz] +                               // [cm^3] * [b] * [1/b-cm]
                                    mesh->Ayz[ie*quad->angleCount()*mesh->yElemCt*mesh->zElemCt + iang*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz] +            // [cm^2]
                                    mesh->Axz[ie*quad->angleCount()*mesh->xElemCt*mesh->zElemCt + iang*mesh->xElemCt*mesh->zElemCt + ix*mesh->zElemCt + iz] +
                                    mesh->Axy[ie*quad->angleCount()*mesh->xElemCt*mesh->yElemCt + iang*mesh->xElemCt*mesh->yElemCt + ix*mesh->yElemCt + iy];

                            //   [#/cm^2] = [#]  / [cm^2]
                            float angFlux = numer/denom;

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

                            angularFlux[ie*ejmp + iang*ajmp + ix*xjmp + iy*yjmp + iz] = angFlux;

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

                float sm = 0.0f;
                for(unsigned int i = 0; i < tempFlux.size(); i++)
                    sm += tempFlux[i];

                for(unsigned int i = 0; i < tempFlux.size(); i++)
                {
                    //int indx = ie*m_mesh->voxelCount() + i; // TODO - delete
                    (*scalarFlux)[ie*mesh->voxelCount() + i] = tempFlux[i];
                }
                emit signalNewIteration(scalarFlux);

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

    emit signalSolverFinished(scalarFlux);
}

void Solver::raytraceHarmonic(const Quadrature *quad, const Mesh *mesh, const XSection *xs, , const int pn)
{
    std::clock_t startMoment = std::clock();

    int groups = xs->groupCount();

    const unsigned short DIRECTION_X = 1;
    const unsigned short DIRECTION_Y = 2;
    const unsigned short DIRECTION_Z = 3;

    std::vector<float> *uflux = new std::vector<float>;
    uflux->resize(groups * mesh->voxelCount());

    unsigned int ejmp = mesh->voxelCount();
    unsigned int xjmp = mesh->xjmp();
    unsigned int yjmp = mesh->yjmp();

    qDebug() << "Running raytracer";

    float tiny = 1.0E-35f;
    float huge = 1.0E35f;
    std::vector<float> meanFreePaths;
    meanFreePaths.resize(xs->groupCount());

    float sx = 25.3906f;
    float sy = 50.0f - 46.4844f;
    float sz = 6.8906f;
    //                                0  1  2  3  4  5  6  7  8  9  0  1  2  3  4  5  6  7  8
    std::vector<float> srcStrength = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};

    unsigned int srcIndxX = int(sx / mesh->dx[0]);  // TODO: This will only work for evenly spaced grids
    unsigned int srcIndxY = int(sy / mesh->dy[0]);
    unsigned int srcIndxZ = int(sz / mesh->dz[0]);

    for(unsigned int zIndxStart = 0; zIndxStart < mesh->zElemCt; zIndxStart++)
        for(unsigned int yIndxStart = 0; yIndxStart < mesh->yElemCt; yIndxStart++)
            for(unsigned int xIndxStart = 0; xIndxStart < mesh->xElemCt; xIndxStart++)  // For every voxel
            {
                float x = mesh->xNodes[xIndxStart] + mesh->dx[xIndxStart]/2;
                float y = mesh->yNodes[yIndxStart] + mesh->dy[yIndxStart]/2;
                float z = mesh->zNodes[zIndxStart] + mesh->dz[zIndxStart]/2;

                std::vector<float> tmpdistv;
                std::vector<float> tmpxsv;
                std::vector<float> mfpv;

                if(xIndxStart == srcIndxX && yIndxStart == srcIndxY && zIndxStart == srcIndxZ)  // End condition
                {
                    float srcToCellDist = sqrt((x-sx)*(x-sx) + (y-sy)*(y-sy) + (z-sz)*(z-sz));
                    unsigned int zid = mesh->zoneId[xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart];
                    float xsval;
                    for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                    {
                        xsval = xs->m_tot1d[zid*groups + ie] * mesh->atomDensity[xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart];
                        (*uflux)[ie*ejmp + xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart] = srcStrength[ie] * exp(-xsval*srcToCellDist) / (4 * M_PI * srcToCellDist * srcToCellDist);
                    }
                    continue;
                }

                // Start raytracing through the geometry
                unsigned int xIndx = xIndxStart;
                unsigned int yIndx = yIndxStart;
                unsigned int zIndx = zIndxStart;

                float srcToCellX = sx - x;
                float srcToCellY = sy - y;
                float srcToCellZ = sz - z;

                float srcToCellDist = sqrt(srcToCellX*srcToCellX + srcToCellY*srcToCellY + srcToCellZ*srcToCellZ);

                float xcos = srcToCellX/srcToCellDist;  // Fraction of direction biased in x-direction, unitless
                float ycos = srcToCellY/srcToCellDist;
                float zcos = srcToCellZ/srcToCellDist;

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
                    float tx = (fabs(xcos) < tiny ? huge : (mesh->xNodes[xBoundIndx] - x)/xcos);  // Distance traveled [cm] when next cell is
                    float ty = (fabs(ycos) < tiny ? huge : (mesh->yNodes[yBoundIndx] - y)/ycos);  //   entered traveling in x direction
                    float tz = (fabs(zcos) < tiny ? huge : (mesh->zNodes[zBoundIndx] - z)/zcos);

                    // Determine the shortest distance traveled [cm] before _any_ surface is crossed
                    float tmin;
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
                    tmpdistv.push_back(tmin);
                    tmpxsv.push_back(xs->m_tot1d[zid*groups + 18] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx]);
                    float gain = tmin * xs->m_tot1d[zid*groups + 18] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx];
                    mfpv.push_back(gain);

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
                        y = mesh->yNodes[yBoundIndx];
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
                        z = mesh->zNodes[zBoundIndx];
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

                    if(xIndx == srcIndxX && yIndx == srcIndxY && zIndx == srcIndxZ)
                    {
                        float finalDist = sqrt((x-sx)*(x-sx) + (y-sy)*(y-sy) + (z-sz)*(z-sz));

                        for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                        {
                            //       [#]       = [cm] * [b] * [1/cm-b]
                            meanFreePaths[ie] += finalDist * xs->m_tot1d[zid*groups + ie] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx];
                        }

                        tmpdistv.push_back(finalDist);
                        tmpxsv.push_back(xs->m_tot1d[zid*groups + 18] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx]);

                        float gain = finalDist * xs->m_tot1d[zid*groups + 18] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx];
                        mfpv.push_back(gain);

                        exhaustedRay = true;
                    }

                } // End of while loop

                for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                {
                    float flx = srcStrength[ie] * exp(-meanFreePaths[ie]) / (4 * M_PI * srcToCellDist * srcToCellDist);

                    if(flx < 0)
                        qDebug() << "solver.cpp: (291): Negative?";

                    if(flx > 1E6)
                        qDebug() << "solver.cpp: (294): Too big!";

                    (*uflux)[ie*ejmp + xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart] = flx;  //srcStrength * exp(-meanFreePaths[ie]) / (4 * M_PI * srcToCellDist * srcToCellDist);
                }

            } // End of each voxel

    qDebug() << "Time to complete raytracer: " << (std::clock() - startMoment)/(double)(CLOCKS_PER_SEC/1000) << " ms";

    emit signalRaytracerFinished(uflux);
    emit signalNewIteration(uflux);
}


void Solver::gsSolverHarmonic(const Quadrature *quad, const Mesh *mesh, const XSection *xs, , const int pn, const std::vector<float> *uflux)
{

    std::clock_t startMoment = std::clock();

    const int maxIterations = 25;
    const float epsilon = 0.01f;


    std::vector<float> angularFlux(xs->groupCount() * quad->angleCount() * mesh->voxelCount());
    std::vector<float> *scalarFlux = new std::vector<float>(xs->groupCount() * mesh->voxelCount(), 0.0f);
    std::vector<float> tempFlux(mesh->voxelCount());
    std::vector<float> preFlux(mesh->voxelCount(), -100.0f);
    std::vector<float> totalSource(mesh->voxelCount(), -100.0f);
    std::vector<float> outboundFluxX(mesh->voxelCount(), -100.0f);
    std::vector<float> outboundFluxY(mesh->voxelCount(), -100.0f);
    std::vector<float> outboundFluxZ(mesh->voxelCount(), -100.0f);
    std::vector<float> extSource(xs->groupCount() * mesh->voxelCount(), 0.0f);

    std::vector<float> errMaxList;
    std::vector<std::vector<float> > errList;
    std::vector<int> converganceIters;
    std::vector<float> converganceTracker;

    errMaxList.resize(xs->groupCount());
    errList.resize(xs->groupCount());
    converganceIters.resize(xs->groupCount());
    converganceTracker.resize(xs->groupCount());

    const XSection &xsref = *xs;

    float influxX = 0.0f;
    float influxY = 0.0f;
    float influxZ = 0.0f;

    int ejmp = mesh->voxelCount() * quad->angleCount();
    int ajmp = mesh->voxelCount();
    int xjmp = mesh->xjmp();
    int yjmp = mesh->yjmp();

    bool downscatterFlag = false;

    if(uflux != NULL)
    {
        qDebug() << "Loading uncollided flux into external source";
        // If there is an uncollided flux provided, use it, otherwise, calculate the external source
        for(unsigned int ei = 0; ei < xs->groupCount(); ei++)
            for(unsigned int ri = 0; ri < mesh->voxelCount(); ri++)
                //                              [#]   =                        [#/cm^2]      * [cm^3]        *  [b]                               * [1/b-cm]
                extSource[ei*mesh->voxelCount() + ri] = (*uflux)[ei*mesh->voxelCount() + ri] * mesh->vol[ri] * xs->scatXs1d(mesh->zoneId[ri], ei) * mesh->atomDensity[ri];

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

    qDebug() << "Solver::gssolver(): 379: Solving " << mesh->voxelCount() * quad->angleCount() * xs->groupCount() << " elements in phase space";

    for(unsigned int ie = 0; ie < xs->groupCount(); ie++)  // for every energy group
    {
        if(!downscatterFlag)
        {
            float dmax = 0.0;
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

        int iterNum = 1;
        float maxDiff = 1.0;

        while(iterNum <= maxIterations && maxDiff > epsilon)  // while not converged
        {
            qDebug() << "Iteration #" << iterNum;

            preFlux = tempFlux;  // Store flux for previous iteration

            for(unsigned int i = 0; i < totalSource.size(); i++)
                totalSource[i] = 0;

            // Calculate the scattering source
            for(unsigned int iie = 0; iie <= ie; iie++)
                for(int iik = 0; iik < (signed) mesh->zElemCt; iik++)
                    for(int iij = 0; iij < (signed)mesh->yElemCt; iij++)
                        for(int iii = 0; iii < (signed)mesh->xElemCt; iii++)
                        {
                            int indx = iii*xjmp + iij*yjmp + iik;
                            int zidIndx = mesh->zoneId[indx];

                            //         [#]    +=  [#]          *      [#/cm^2]                               * [b]                          * [1/b-cm]                * [cm^3]
                            totalSource[indx] += 1.0/(4.0*M_PI)*(*scalarFlux)[iie*mesh->voxelCount() + indx] * xsref.scatXs1d(zidIndx, iie) * mesh->atomDensity[indx] * mesh->vol[indx]; //xsref(ie-1, zidIndx, 0, iie));
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

            for(int iang = 0; iang < quad->angleCount(); iang++)  // for every angle
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

                            float numer = totalSource[ix*xjmp+iy*yjmp+iz] +                                                                                              // [#]
                                    mesh->Ayz[ie*quad->angleCount()*mesh->yElemCt*mesh->zElemCt + iang*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz] * influxX +  // [cm^2 * #/cm^2]  The 2x is already factored in
                                    mesh->Axz[ie*quad->angleCount()*mesh->xElemCt*mesh->zElemCt + iang*mesh->xElemCt*mesh->zElemCt + ix*mesh->zElemCt + iz] * influxY +
                                    mesh->Axy[ie*quad->angleCount()*mesh->xElemCt*mesh->yElemCt + iang*mesh->xElemCt*mesh->yElemCt + ix*mesh->yElemCt + iy] * influxZ;
                            float denom = mesh->vol[ix*xjmp+iy*yjmp+iz]*xsref.totXs1d(zid, ie)*mesh->atomDensity[ix*xjmp + iy*yjmp + iz] +                               // [cm^3] * [b] * [1/b-cm]
                                    mesh->Ayz[ie*quad->angleCount()*mesh->yElemCt*mesh->zElemCt + iang*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz] +            // [cm^2]
                                    mesh->Axz[ie*quad->angleCount()*mesh->xElemCt*mesh->zElemCt + iang*mesh->xElemCt*mesh->zElemCt + ix*mesh->zElemCt + iz] +
                                    mesh->Axy[ie*quad->angleCount()*mesh->xElemCt*mesh->yElemCt + iang*mesh->xElemCt*mesh->yElemCt + ix*mesh->yElemCt + iy];

                            //   [#/cm^2] = [#]  / [cm^2]
                            float angFlux = numer/denom;

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

                            angularFlux[ie*ejmp + iang*ajmp + ix*xjmp + iy*yjmp + iz] = angFlux;

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

                float sm = 0.0f;
                for(unsigned int i = 0; i < tempFlux.size(); i++)
                    sm += tempFlux[i];

                for(unsigned int i = 0; i < tempFlux.size(); i++)
                {
                    //int indx = ie*m_mesh->voxelCount() + i; // TODO - delete
                    (*scalarFlux)[ie*mesh->voxelCount() + i] = tempFlux[i];
                }
                emit signalNewIteration(scalarFlux);

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

    emit signalNewIteration(scalarFlux);
    emit signalSolverFinished(scalarFlux);
}
