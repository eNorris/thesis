#include "solver.h"

//#include "solvers.h"
#define _USE_MATH_DEFINES
#include <cmath>

//#include "mainwindow.h"

#include <QDebug>
#include <iostream>

#include <ctime>

#include "quadrature.h"
#include "mesh.h"
#include "xsection.h"
#include "outwriter.h"

#include "gui/outputdialog.h"

Solver::Solver(QObject *parent) : QObject(parent)
{

}

Solver::~Solver()
{

}



//std::vector<float> Solver::raytrace(const Quadrature *quad, const Mesh *mesh, const XSection *xs)
void Solver::raytrace(const Quadrature *quad, const Mesh *mesh, const XSection *xs)
{
    std::clock_t startMoment = std::clock();

    int groups = xs->groupCount();

    //const unsigned short DIRECTION_NONE = 0;
    const unsigned short DIRECTION_X = 1;
    const unsigned short DIRECTION_Y = 2;
    const unsigned short DIRECTION_Z = 3;

    std::vector<float> uflux;
    uflux.resize(groups * mesh->voxelCount());

    unsigned int ejmp = mesh->voxelCount();
    unsigned int xjmp = mesh->xjmp();
    unsigned int yjmp = mesh->yjmp();

    qDebug() << "Running raytracer";

    float tiny = 1.0E-35;
    float huge = 1.0E35;
    //float e3 = 1.0E-8;
    //float e4 = 2.5E-9;
    std::vector<float> meanFreePaths;
    meanFreePaths.resize(xs->groupCount());

    //for(unsigned int is = 0; is < config->sourceIntensity.size(); is++)
    //{
    float sx = 10.0f;  //config->sourceX[is];
    float sy = 10.0f;  //config->sourceY[is];
    float sz = 10.0f;  //config->sourceZ[is];
    float srcStrength = 1.0f;
    //float srcStrength = config->sourceIntensity[is];

    unsigned int srcIndxX = int(sx / mesh->dx[0]);  // TODO: This will only work for evenly spaced grids
    unsigned int srcIndxY = int(sy / mesh->dy[0]);
    unsigned int srcIndxZ = int(sz / mesh->dz[0]);

    //for(unsigned int xIndxStart = 0; xIndxStart < mesh->xElemCt; xIndxStart++)
    //    for(unsigned int yIndxStart = 0; yIndxStart < mesh->yElemCt; yIndxStart++)
    //        for(unsigned int zIndxStart = 0; zIndxStart < mesh->zElemCt; zIndxStart++)
    for(unsigned int zIndxStart = 0; zIndxStart < mesh->zElemCt; zIndxStart++)
        for(unsigned int yIndxStart = 0; yIndxStart < mesh->yElemCt; yIndxStart++)
            for(unsigned int xIndxStart = 0; xIndxStart < mesh->xElemCt; xIndxStart++)
            {
                float x = mesh->xNodes[xIndxStart] + mesh->dx[xIndxStart]/2;
                float y = mesh->yNodes[yIndxStart] + mesh->dy[yIndxStart]/2;
                float z = mesh->zNodes[zIndxStart] + mesh->dz[zIndxStart]/2;

                if(xIndxStart == 5 && yIndxStart == 52 && zIndxStart == 13)
                {
                    qDebug() << "Halt!!!!";
                }


                if(xIndxStart == srcIndxX && yIndxStart == srcIndxY && zIndxStart == srcIndxZ)
                {
                    // TODO: Calculate
                    float srcToCellDist = sqrt((x-sx)*(x-sx) + (y-sy)*(y-sy) + (z-sz)*(z-sz));
                    unsigned int zid = mesh->zoneId[xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart];
                    float xsval;
                    for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                    {
                        xsval = xs->m_tot1d[zid*groups + ie] * mesh->atomDensity[xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart];
                        uflux[ie*ejmp + xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart] = srcStrength * exp(-xsval*srcToCellDist) / (4 * M_PI * srcToCellDist * srcToCellDist);
                    }
                    continue;
                }

                int xIndx = xIndxStart;
                int yIndx = yIndxStart;
                int zIndx = zIndxStart;

                // TODO: Do these need to reverse direction? Right now it's source -> cell
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

                //int mchk = 1;
                bool exhaustedRay = false;
                while(!exhaustedRay)
                {
                    // Determine the distance to cell boundaries
                    //float zz = fabs(xcos);
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

                    // TODO - handle this a little smarter
                    if(tmin < -3E-8)  // Include a little padding for hitting an edge
                        qDebug() << "Reversed space!";

                    // Update mpf array
                    unsigned int zid = mesh->zoneId[xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart];
                    for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                    {
                        //xsval = xs->m_tot1d[zid*groups + ie] * mesh->atomDensity[xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart];
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
                        }
                        else
                        {
                            xIndx--;
                            xBoundIndx--;
                        }
                        //if(xIndx < 0 || xIndx >= (signed) mesh->xElemCt)
                        //    exhaustedRay = true;
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
                        //if(yIndx < 0 || yIndx >= (signed) mesh->yElemCt)
                        //    exhaustedRay = true;
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
                        //if(zIndx < 0 || zIndx >= (signed) mesh->zElemCt)
                        //    exhaustedRay = true;
                    }

                    if(xIndx == srcIndxX && yIndx == srcIndxY && zIndx == srcIndxZ)
                    {
                        //meanFreePaths[ie] += tmin * xs->m_tot1d[zid*groups + ie] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx];

                        float finalDist = sqrt((x-sx)*(x-sx) + (y-sy)*(y-sy) + (z-sz)*(z-sz));
                        //unsigned int zid = mesh->zoneId[xIndx*xjmp + yIndx*yjmp + zIndx];
                        //float xsval;
                        for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                        {
                            meanFreePaths[ie] += finalDist * xs->m_tot1d[zid*groups + ie] * mesh->atomDensity[xIndx*xjmp + yIndx*yjmp + zIndx];
                            //xsval = xs->m_tot1d[zid*groups + ie] * mesh->atomDensity[xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart];
                            //uflux[ie*ejmp + xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart] = srcStrength * exp(-xsval*srcToCellDist) / (4 * M_PI * srcToCellDist * srcToCellDist);
                        }

                        exhaustedRay = true;
                    }

                } // End of while loop

                for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                {
                    // TODO - For now all sources emit the same strength for all energies
                    //unsigned short zid = mesh->zoneId[xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart];
                    //float optical = meanFreePaths[ie];
                    //float mfp = meanFreePaths[ie];
                    //float r2 = 4 * M_PI * srcToCellDist * srcToCellDist;
                    //float uf = srcStrength * exp(-optical) / r2;

                    //float mfp = meanFreePaths[ie];
                    float flx = srcStrength * exp(-meanFreePaths[ie]) / (4 * M_PI * srcToCellDist * srcToCellDist);

                    if(flx < 0)
                        qDebug() << "raytracer.cpp: (223): Negative?";

                    if(flx > 1E6)
                        qDebug() << "raytracer.cpp: (226): Too big!";

                    uflux[ie*ejmp + xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart] = flx;  //srcStrength * exp(-meanFreePaths[ie]) / (4 * M_PI * srcToCellDist * srcToCellDist);
                }

            } // End of each voxel
    //}

    qDebug() << "Time to complete raytracer: " << (std::clock() - startMoment)/(double)(CLOCKS_PER_SEC/1000) << " ms";

    // TODO - should return a pointer!
    emit signalNewIteration(&uflux);
    //return uflux;
}




//std::vector<float> Solver::gssolver(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const std::vector<float> *uflux)
void Solver::gssolver(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const std::vector<float> *uflux)
{

    std::clock_t startMoment = std::clock();

    const int maxIterations = 25;
    const float epsilon = 0.01;


    std::vector<float> angularFlux(xs->groupCount() * quad->angleCount() * mesh->voxelCount(), 0.0f);
    std::vector<float> *scalarFlux = new std::vector<float>(xs->groupCount() * mesh->voxelCount(), 0.0f);
    std::vector<float> tempFlux(mesh->voxelCount());
    std::vector<float> preFlux(mesh->voxelCount(), 0.0f);
    std::vector<float> totalSource(mesh->voxelCount(), 0.0f);
    std::vector<float> outboundFluxX(mesh->voxelCount(), 0.0f);
    std::vector<float> outboundFluxY(mesh->voxelCount(), 0.0f);
    std::vector<float> outboundFluxZ(mesh->voxelCount(), 0.0f);
    std::vector<float> extSource(xs->groupCount() * mesh->voxelCount(), 0.0f);

    std::vector<float> errMaxList;
    std::vector<std::vector<float> > errList;
    std::vector<int> converganceIters;

    errMaxList.resize(xs->groupCount());
    errList.resize(xs->groupCount());
    converganceIters.resize(xs->groupCount());

    /*
    std::vector<float> angularFlux;
    std::vector<float> scalarFlux;
    std::vector<float> tempFlux;
    std::vector<float> preFlux;
    std::vector<float> totalSource;
    std::vector<float> errList;
    std::vector<float> outboundFluxX;
    std::vector<float> outboundFluxY;
    std::vector<float> outboundFluxZ;
    std::vector<float> extSource;
    */

    const XSection &xsref = *xs;

    float influxX = 0.0f;
    float influxY = 0.0f;
    float influxZ = 0.0f;

    int ejmp = mesh->voxelCount() * quad->angleCount();
    int ajmp = mesh->voxelCount();
    int xjmp = mesh->xjmp();
    int yjmp = mesh->yjmp();

    /*
    angularFlux.resize(xs->groupCount() * quad->angleCount() * mesh->voxelCount());
    scalarFlux.resize(xs->groupCount() * mesh->voxelCount(), 0.0f);
    tempFlux.resize(mesh->voxelCount());
    outboundFluxX.resize(mesh->voxelCount(), 0.0f);
    outboundFluxY.resize(mesh->voxelCount(), 0.0f);
    outboundFluxZ.resize(mesh->voxelCount(), 0.0f);
    totalSource.resize(mesh->voxelCount(), 0.0f);
    isocaSource.resize(mesh->voxelCount(), 0.0f);
    */

    //extSource.resize(xs->groupCount() * mesh->voxelCount(), 0.0f);

    if(uflux != NULL)
    {
        qDebug() << "Loading uncollided flux into external source";
        // If there is an uncollided flux provided, use it, otherwise, calculate the external source
        //for(unsigned int i = 0; i < uflux->size(); i++)
        for(unsigned int ei = 0; ei < xs->groupCount(); ei++)
            for(unsigned int ri = 0; ri < mesh->voxelCount(); ri++)
                extSource[ei*mesh->voxelCount() + ri] = (*uflux)[ei*mesh->voxelCount() + ri] * mesh->vol[ri] * xs->scatXs1d(mesh->zoneId[ri], ei);
    }
    else
    {
        qDebug() << "Building external source";
        // Calculate the external source mesh
        // TODO should be a function of energy
        for(unsigned int ei = 0; ei < xs->groupCount(); ei++)
        {
            int srcIndxX = mesh->xElemCt/2;
            int srcIndxY = mesh->yElemCt/2;
            int srcIndxZ = mesh->zElemCt/2;
            extSource[ei * mesh->voxelCount() + srcIndxX*xjmp + srcIndxY*yjmp + srcIndxZ] = 1.0;  //config->sourceIntensity[is];
        }
    }

    //extSource[((mesh->xMesh - 1)/2)*xjmp + ((mesh->yMesh-1)/2)*yjmp + ((mesh->zMesh-1)/2)] = 1E6;  // [gammas / sec]
    //extSource[((mesh->xElemCt - 1)/2)*xjmp + (config->colYLen/2)*yjmp + ((mesh->zElemCt-1)/2)] = 1E6;

    qDebug() << "Solving " << mesh->voxelCount() * quad->angleCount() * xs->groupCount() << " elements in phase space";

    for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
    {
        qDebug() << "Energy group #" << ie;
        // Do the solving...

        int iterNum = 1;
        float maxDiff = 1.0;

        while(iterNum <= maxIterations && maxDiff > epsilon)
        {
            qDebug() << "Iteration #" << iterNum;

            preFlux = tempFlux;  // Store flux for previous iteration

            for(unsigned int i = 0; i < totalSource.size(); i++)
                totalSource[i] = 0;
                //isocaSource[i] = 0;

            // Calculate the scattering source
            for(unsigned int iie = 0; iie <= ie; iie++)
                for(int iik = 0; iik < (signed) mesh->zElemCt; iik++)
                    for(int iij = 0; iij < (signed)mesh->yElemCt; iij++)
                        for(int iii = 0; iii < (signed)mesh->xElemCt; iii++)
                        {
                            int indx = iii*xjmp + iij*yjmp + iik;
                            int zidIndx = mesh->zoneId[indx];

                            totalSource[indx] += 1.0/(4.0*M_PI)*(*scalarFlux)[iie*mesh->voxelCount() + indx] * xsref.scatXs1d(zidIndx, iie) * mesh->vol[indx]; //xsref(ie-1, zidIndx, 0, iie));
                            //isocaSource[indx] += 1.0/(4.0*M_PI)*scalarFlux[iie*mesh->voxelCount() + indx] * xsref.scatXs1d(zidIndx, iie) * mesh->vol[indx]; //xsref(ie-1, zidIndx, 0, iie));
                        }



            //float isomax = -1;
            //for(unsigned int ti = 0; ti < isocaSource.size(); ti++)
            //    if(isocaSource[ti] > isomax)
            //        isomax = isocaSource[ti];

            //float scamax = -1;
            //for(unsigned int ti = 0; ti < scalarFlux.size(); ti++)
            //    if(scalarFlux[ti] > scamax)
            //        scamax = scalarFlux[ti];

            // Calculate the total source
            for(unsigned int ri = 0; ri < mesh->voxelCount(); ri++)
                totalSource[ri] += extSource[ie*mesh->voxelCount() + ri];

            float s = 0;
            for(unsigned int ri = 0; ri < mesh->voxelCount(); ri++)
                s += totalSource[ri];

            qDebug() << "S = " << s;

            // Clear for a new sweep
            for(unsigned int i = 0; i < tempFlux.size(); i++)
                tempFlux[i] = 0;

            for(int iang = 0; iang < quad->angleCount(); iang++)
            {
                qDebug() << "Angle #" << iang;

                int izStart = 0;  // Sweep start index
                int diz = 1;  // Sweep direction
                if(quad->eta[iang] < 0)  // Condition to sweep backward
                {
                    izStart = mesh->zElemCt - 1;  // Start at the far end
                    diz = -1;  // Sweep toward zero
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

                //for(int iz = 0; iz < mesh->zMesh; iz++)  // Start at origin and sweep towards corner
                //for(; iz < (signed) mesh->zElemCt && iz >= 0; iz += diz)
                int iz = izStart;
                while(iz < (signed) mesh->zElemCt && iz >= 0)
                {
                    //for(int iy = 0; iy < mesh->yMesh; iy++)
                    //for(; iy < (signed) mesh->yElemCt && iy >= 0; iy += diy)
                    //for(int iy = quad->zi[iang] >= 0 ? 0 : mesh->yElemCt-1; quad->zi[iang] >= 0 ? iy < mesh->yElemCt : iy >= 0; iy += (quad->zi[iang] >= 0 ? 1 : -1))
                    int iy = iyStart;
                    while(iy < (signed) mesh->yElemCt && iy >= 0)
                    {
                        //for(int ix = 0; ix < mesh->xMesh; ix++)
                        //for(; ix < (signed) mesh->xElemCt && ix >= 0; ix += dix)
                        //for(int ix = quad->mu[iang] >= 0 ? 0 : mesh->xElemCt-1; quad->mu[iang] >= 0 ? ix < mesh->xElemCt : ix >= 0; ix += (quad->mu[iang] >= 0 ? 1 : -1))
                        int ix = ixStart;
                        while(ix < (signed) mesh->xElemCt && ix >= 0)
                        {

                            //qDebug() << ix << "   " << iy << "   " << iz;

                            //if(ix == 128 && iy == 128 && iz == 64)
                            //    qDebug() << "Found the one data point!";

                            int zid = mesh->zoneId[ix*xjmp + iy*yjmp + iz];

                            // Handle the x influx
                            if(quad->mu[iang] >= 0) // Approach x = 0 -> xMesh
                            {
                                if(ix == 0)
                                    influxX = 0.0f;
                                else
                                    influxX = outboundFluxX[(ix-1)*xjmp + iy*yjmp + iz];
                                    //influxX = angularFlux[ie*ejmp + iang*ajmp + (ix-1)*xjmp + iy*yjmp + iz];
                            }
                            else // Approach x = xMesh-1 -> 0
                            {
                                if(ix == (signed) mesh->xElemCt-1)
                                    influxX = 0.0f;
                                else
                                    influxX = outboundFluxX[(ix+1)*xjmp + iy*yjmp + iz];
                                //if(influxX != 0)
                                //    qDebug() << "cat";
                                    //influxX = angularFlux[ie*ejmp + iang*ajmp + (ix+1)*xjmp + iy*yjmp + iz];
                            }

                            // Handle the y influx
                            if(quad->zi[iang] >= 0) // Approach y = 0 -> yMesh
                            {
                                if(iy == 0)
                                    influxY = 0.0f;
                                else
                                    influxY = outboundFluxY[ix*xjmp + (iy-1)*yjmp + iz];
                                    //influxY = angularFlux[ie*ejmp + iang*ajmp + ix*xjmp + (iy-1)*yjmp + iz];
                            }
                            else // Approach y = yMesh-1 -> 0
                            {
                                if(iy == (signed) mesh->yElemCt-1)
                                    influxY = 0.0f;
                                else
                                    influxY = outboundFluxY[ix*xjmp + (iy+1)*yjmp + iz];
                                    //influxY = angularFlux[ie*ejmp + iang*ajmp + ix*xjmp + (iy+1)*yjmp + iz];
                            }

                            // Handle the z influx
                            if(quad->eta[iang] >= 0)
                            {
                                if(iz == 0)
                                    influxZ = 0.0f;
                                else
                                    influxZ = outboundFluxZ[ix*xjmp + iy*yjmp + iz-1];
                                    //influxZ = angularFlux[ie*ejmp + iang*ajmp + ix*xjmp + iy*yjmp + iz-1];
                            }
                            else
                            {
                                if(iz == (signed) mesh->zElemCt-1)
                                    influxZ = 0.0f;
                                else
                                    influxZ = outboundFluxZ[ix*xjmp + iy*yjmp + iz+1];
                                    //influxZ = angularFlux[ie*ejmp + iang*ajmp + ix*xjmp + iy*yjmp + iz+1];
                            }

                            // I don't think the *vol should be here
                            // I'm pretty sure totalSource isn't normalized by volume...
                            float numer = totalSource[ix*xjmp+iy*yjmp+iz] + //* mesh->vol[ix*xjmp+iy*yjmp+iz] +
                                    mesh->Ayz[ie*quad->angleCount()*mesh->yElemCt*mesh->zElemCt + iang*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz] * influxX +  // The 2x is already factored in
                                    mesh->Axz[ie*quad->angleCount()*mesh->xElemCt*mesh->zElemCt + iang*mesh->xElemCt*mesh->zElemCt + ix*mesh->zElemCt + iz] * influxY +
                                    mesh->Axy[ie*quad->angleCount()*mesh->xElemCt*mesh->yElemCt + iang*mesh->xElemCt*mesh->yElemCt + ix*mesh->yElemCt + iy] * influxZ;
                            float denom = mesh->vol[ix*xjmp+iy*yjmp+iz]*xsref.totXs1d(zid, ie) +   //xsref(ie, zid, 0, 0) +  //xs->operator()(ie, zid, 0, 0) +
                                    mesh->Ayz[ie*quad->angleCount()*mesh->yElemCt*mesh->zElemCt + iang*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz] +  // The 2x is already factored in
                                    mesh->Axz[ie*quad->angleCount()*mesh->xElemCt*mesh->zElemCt + iang*mesh->xElemCt*mesh->zElemCt + ix*mesh->zElemCt + iz] +
                                    mesh->Axy[ie*quad->angleCount()*mesh->xElemCt*mesh->yElemCt + iang*mesh->xElemCt*mesh->yElemCt + ix*mesh->yElemCt + iy];

                            //if(numer > 0)
                            //    qDebug() << "Caught numer";

                            float angFlux = numer/denom;

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

                            //if(angFlux != 0)
                            //    qDebug() << "got it";
                            //if(influxX > 0 || influxY > 0 || influxZ > 0)
                            //    qDebug() << "Gotcha!";

                            outboundFluxX[ix*xjmp + iy*yjmp + iz] = 2*angFlux - influxX;
                            outboundFluxY[ix*xjmp + iy*yjmp + iz] = 2*angFlux - influxY;
                            outboundFluxZ[ix*xjmp + iy*yjmp + iz] = 2*angFlux - influxZ;

                            //float outX = 2*angFlux - influxX;
                            //float outY = 2*angFlux - influxY;
                            //float outZ = 2*angFlux - influxZ;


                            if(outboundFluxX[ix*xjmp + iy*yjmp + iz] < 0)
                                outboundFluxX[ix*xjmp + iy*yjmp + iz] = 0;

                            if(outboundFluxY[ix*xjmp + iy*yjmp + iz] < 0)
                                outboundFluxY[ix*xjmp + iy*yjmp + iz] = 0;

                            if(outboundFluxZ[ix*xjmp + iy*yjmp + iz] < 0)
                                outboundFluxZ[ix*xjmp + iy*yjmp + iz] = 0;



                            //if(denom == 0)
                            //    qDebug() << "All life is over!";

                            //if(numer != 0)
                            //    qDebug() << "Got a fish!";

                            // Sum all the angular fluxes
                            tempFlux[ix*xjmp + iy*yjmp + iz] += quad->wt[iang]*angularFlux[ie*ejmp + iang*ajmp + ix*xjmp + iy*yjmp + iz];

                            ix += dix;
                        } // end of for ix

                        iy += diy;
                    } // end of for iy

                    iz += diz;
                } // end of for iz

                float sm = 0.0f;
                for(int i = 0; i < tempFlux.size(); i++)
                    sm += tempFlux[i];

                //if(outputDialog->debuggingEnabled())
                //{
                //    emit signalDebugHalt(tempFlux);
                //    m_pendingUserContinue.wait(&m_mutex);
                //}


            } // end of all angles

            maxDiff = -1E35;
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
                //int indx = ie*m_mesh->voxelCount() + i; // TODO - delete
                (*scalarFlux)[ie*mesh->voxelCount() + i] = tempFlux[i];
            }

            iterNum++;
            emit signalNewIteration(scalarFlux);
        } // end not converged

        //emit signalNewIteration(scalarFlux);
    }  // end each energy group

    qDebug() << "Time to complete: " << (std::clock() - startMoment)/(double)(CLOCKS_PER_SEC/1000) << " ms";

    //OutWriter::writeScalarFluxMesh("./outlog.matmsh", *m_mesh, scalarFlux);

    for(unsigned int i = 0; i < errList.size(); i++)
    {
        qDebug() << "Group: " << i << "   maxDiff: " << errMaxList[i];
        qDebug() << "Iterations: " << converganceIters[i];
        for(int j = 0; j < errList[i].size(); j++)
            std::cout << errList[i][j] << "\t";
        std::cout << "\n" << std::endl;
    }

    //return *scalarFlux;
}
