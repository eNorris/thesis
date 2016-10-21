//#include "solvers.h"
/*
#define _USE_MATH_DEFINES
#include <cmath>

#include "mainwindow.h"

#include <QDebug>
#include <iostream>

#include <ctime>

#include "quadrature.h"
#include "mesh.h"
#include "xsection.h"
#include "config.h"
#include "outwriter.h"

#include "gui/outputdialog.h"


std::vector<float> MainWindow::gssolver(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const std::vector<float> *uflux)
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

    const XSection &xsref = *xs;

    float influxX = 0.0f;
    float influxY = 0.0f;
    float influxZ = 0.0f;

    int ejmp = mesh->voxelCount() * quad->angleCount();
    int ajmp = mesh->voxelCount();
    int xjmp = mesh->xjmp();
    int yjmp = mesh->yjmp();


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
                            float numer = totalSource[ix*xjmp+iy*yjmp+iz] + // * mesh->vol[ix*xjmp+iy*yjmp+iz] +
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


                            if(outboundFluxX[ix*xjmp + iy*yjmp + iz] < 0)
                                outboundFluxX[ix*xjmp + iy*yjmp + iz] = 0;

                            if(outboundFluxY[ix*xjmp + iy*yjmp + iz] < 0)
                                outboundFluxY[ix*xjmp + iy*yjmp + iz] = 0;

                            if(outboundFluxZ[ix*xjmp + iy*yjmp + iz] < 0)
                                outboundFluxZ[ix*xjmp + iy*yjmp + iz] = 0;

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

                if(outputDialog->debuggingEnabled())
                {
                    emit signalDebugHalt(tempFlux);
                    m_pendingUserContinue.wait(&m_mutex);
                }


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
                (*scalarFlux)[ie*m_mesh->voxelCount() + i] = tempFlux[i];
            }

            iterNum++;
        } // end not converged

        emit signalNewIteration(scalarFlux);
    }  // end each energy group

    qDebug() << "Time to complete: " << (std::clock() - startMoment)/(double)(CLOCKS_PER_SEC/1000) << " ms";

    //OutWriter::writeScalarFluxMesh("./outlog.matmsh", *m_mesh, scalarFlux);

    for(unsigned int i = 0; i < errList.size(); i++)
    {
        qDebug() << "Group: " << i << "   maxDiff: " << errMaxList[i];
        qDebug() << "Iterations: " << converganceIters[i];
        for(int j = 0; j < errList[i].size(); j++)
            std::cout << errList[i][j] << "\t";
        std::cout << "\n";
    }

    return *scalarFlux;
}
*/
