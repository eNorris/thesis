//#include "solvers.h"
#define _USE_MATH_DEFINES
#include <cmath>

#include "mainwindow.h"

#include <QDebug>

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

    std::vector<float> scalarFlux;
    std::vector<float> angularFlux;
    std::vector<float> tempFlux;
    std::vector<float> preFlux;
    std::vector<float> totalSource;
    std::vector<float> isocaSource;
    std::vector<float> errList;
    std::vector<float> outboundFluxX;
    std::vector<float> outboundFluxY;
    std::vector<float> outboundFluxZ;

    std::vector<float> extSource;

    const XSection &xsref = *xs;

    float influxX = 0.0f;
    float influxY = 0.0f;
    float influxZ = 0.0f;

    int ejmp = mesh->voxelCount() * quad->angleCount();
    int ajmp = mesh->voxelCount();
    int xjmp = mesh->xjmp();
    int yjmp = mesh->yjmp();

    scalarFlux.resize(xs->groupCount() * mesh->voxelCount(), 0.0f);
    tempFlux.resize(mesh->voxelCount());
    outboundFluxX.resize(mesh->voxelCount(), 0.0f);
    outboundFluxY.resize(mesh->voxelCount(), 0.0f);
    outboundFluxZ.resize(mesh->voxelCount(), 0.0f);
    angularFlux.resize(xs->groupCount() * quad->angleCount() * mesh->voxelCount());
    totalSource.resize(mesh->voxelCount(), 0.0f);
    isocaSource.resize(mesh->voxelCount(), 0.0f);

    extSource.resize(mesh->voxelCount(), 0.0f);

    if(uflux != NULL)
    {
        // If there is an uncollided flux provided, use it, otherwise, calculate the external source
        for(unsigned int i = 0; i < uflux->size(); i++)
            extSource[i] = (*uflux)[i] * mesh->vol[i] * xs->scatXs1d(mesh->zoneId[i], 0);
    }
    else
    {
        // Calculate the external source mesh
        // TODO should be a function of energy
        for(unsigned int is = 0; is < xs->groupCount(); is++)
        {
            // Make sure the source is inside the mesh
            //if(config->sourceX[is] < mesh->xNodes[0] || config->sourceX[is] > mesh->xNodes[mesh->xNodeCt-1])
            //    qDebug() << "WARNING: Source " << is << " is outside the x mesh(" << mesh->xNodes[0] << ", " << mesh->xNodes[mesh->xNodeCt-1] << ")!";
            //if(config->sourceY[is] < mesh->yNodes[0] || config->sourceY[is] > mesh->yNodes[mesh->yNodeCt-1])
            //    qDebug() << "WARNING: Source " << is << " is outside the y mesh(" << mesh->yNodes[0] << ", " << mesh->yNodes[mesh->yNodeCt-1] << ")!";
            //if(config->sourceZ[is] < mesh->zNodes[0] || config->sourceZ[is] > mesh->zNodes[mesh->zNodeCt-1])
            //    qDebug() << "WARNING: Source " << is << " is outside the z mesh(" << mesh->zNodes[0] << ", " << mesh->zNodes[mesh->zNodeCt-1] << ")!";

            /*
            int srcIndxX = -1;
            int srcIndxY = -1;
            int srcIndxZ = -1;

            for(unsigned int ix = 1; ix < mesh->xNodeCt; ix++)
                if(mesh->xNodes[ix] > config->sourceX[is])
                {
                    srcIndxX = ix-1;  // Subtract 1 to go from node to element
                    break;
                }

            for(unsigned int iy = 1; iy < mesh->yNodeCt; iy++)
                if(mesh->yNodes[iy] > config->sourceY[is])
                {
                    srcIndxY = iy-1;
                    break;
                }

            for(unsigned int iz = 1; iz < mesh->zNodeCt; iz++)
                if(mesh->zNodes[iz] > config->sourceZ[is])
                {
                    srcIndxZ = iz-1;
                    break;
                }
                */

            //if(srcIndxX == -1 || srcIndxY == -1 || srcIndxZ == -1)
            //    qDebug() << "ERROR: Illegal source location!";

            //extSource[srcIndxX*xjmp + srcIndxY*yjmp + srcIndxZ] = config->sourceIntensity[is];
            int srcIndxX = mesh->xElemCt/2;
            int srcIndxY = mesh->yElemCt/2;
            int srcIndxZ = mesh->zElemCt/2;
            extSource[srcIndxX*xjmp + srcIndxY*yjmp + srcIndxZ] = 1.0;  //config->sourceIntensity[is];
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

            for(unsigned int i = 0; i < isocaSource.size(); i++)
                isocaSource[i] = 0;

            // Calculate the scattering source
            for(unsigned int iie = 0; iie <= ie; iie++)
                for(int iik = 0; iik < (signed) mesh->zElemCt; iik++)
                    for(int iij = 0; iij < (signed)mesh->yElemCt; iij++)
                        for(int iii = 0; iii < (signed)mesh->xElemCt; iii++)
                        {
                            int indx = iii*xjmp + iij*yjmp + iik;
                            int zidIndx = mesh->zoneId[indx];
                            //float xsval = xsref.scatXs(zidIndx, iie, ie);
                            //qDebug() << xsval;
                            isocaSource[indx] += 1.0/(4.0*M_PI)*scalarFlux[iie*mesh->voxelCount() + indx] * xsref.scatXs1d(zidIndx, iie) * mesh->vol[indx]; //xsref(ie-1, zidIndx, 0, iie));
                        }



            float isomax = -1;
            for(unsigned int ti = 0; ti < isocaSource.size(); ti++)
                if(isocaSource[ti] > isomax)
                    isomax = isocaSource[ti];

            float scamax = -1;
            for(unsigned int ti = 0; ti < scalarFlux.size(); ti++)
                if(scalarFlux[ti] > scamax)
                    scamax = scalarFlux[ti];

            // Calculate the total source
            for(unsigned int i = 0; i < mesh->voxelCount(); i++)
                totalSource[i] = extSource[ie*mesh->voxelCount() + i] + isocaSource[i];

            // Clear for a new sweep
            for(unsigned int i = 0; i < tempFlux.size(); i++)
                tempFlux[i] = 0;

            for(int iang = 0; iang < quad->angleCount(); iang++)
            {
                qDebug() << "Angle #" << iang;

                //if(quad->mu[iang] > 0 && quad->zi[iang] > 0  && quad->eta[iang] > 0)  // Octant 1
                //{

                /*
                if(quad->mu[iang] < 0 && quad->zi[iang] < 0 && quad->eta[iang] < 0)
                    continue;

                if(quad->mu[iang] > 0 && quad->zi[iang] < 0 && quad->eta[iang] < 0)
                    continue;

                if(quad->mu[iang] < 0 && quad->zi[iang] > 0 && quad->eta[iang] < 0)
                    continue;

                if(quad->mu[iang] > 0 && quad->zi[iang] > 0 && quad->eta[iang] < 0)
                    continue;

                if(quad->mu[iang] < 0 && quad->zi[iang] < 0 && quad->eta[iang] > 0)
                    continue;

                if(quad->mu[iang] > 0 && quad->zi[iang] < 0 && quad->eta[iang] > 0)
                    continue;

                if(quad->mu[iang] < 0 && quad->zi[iang] > 0 && quad->eta[iang] > 0)
                    continue;
                    */

                //if(quad->mu[iang] > 0 && quad->zi[iang] > 0 && quad->eta[iang] > 0)
                //    continue;
                int iz = 0;
                int diz = 1;
                if(quad->eta[iang] < 0)
                {
                    iz = mesh->zElemCt - 1;
                    diz = -1;
                }

                int iy = 0;
                int diy = 1;
                if(quad->zi[iang] < 0)
                {
                    iy = mesh->yElemCt - 1;
                    diy = -1;
                }

                int ix = 0;
                int dix = 1;
                if(quad->mu[iang] < 0)
                {
                    ix = mesh->xElemCt - 1;
                    dix = -1;
                }

                //for(int iz = 0; iz < mesh->zMesh; iz++)  // Start at origin and sweep towards corner
                for(; iz < (signed) mesh->zElemCt && iz >= 0; iz += diz)
                {
                    //for(int iy = 0; iy < mesh->yMesh; iy++)
                    for(; iy < (signed) mesh->yElemCt && iy >= 0; iy += diy)
                    //for(int iy = quad->zi[iang] >= 0 ? 0 : mesh->yElemCt-1; quad->zi[iang] >= 0 ? iy < mesh->yElemCt : iy >= 0; iy += (quad->zi[iang] >= 0 ? 1 : -1))
                    {
                        //for(int ix = 0; ix < mesh->xMesh; ix++)
                        for(; ix < (signed) mesh->xElemCt && ix >= 0; ix += dix)
                        //for(int ix = quad->mu[iang] >= 0 ? 0 : mesh->xElemCt-1; quad->mu[iang] >= 0 ? ix < mesh->xElemCt : ix >= 0; ix += (quad->mu[iang] >= 0 ? 1 : -1))
                        {
                            int zid = mesh->zoneId[ix*xjmp + iy*yjmp + iz];

                            // Handle the x influx
                            //float t1 = quad->mu[iang];
                            //float t2 = quad->zi[iang];
                            //float t3 = quad->eta[iang];
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
                                    mesh->Ayz[iang*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz] * influxX +  // The 2x is already factored in
                                    mesh->Axz[iang*mesh->xElemCt*mesh->zElemCt + ix*mesh->zElemCt + iz] * influxY +
                                    mesh->Axy[iang*mesh->xElemCt*mesh->yElemCt + ix*mesh->yElemCt + iy] * influxZ;
                            float denom = mesh->vol[ix*xjmp+iy*yjmp+iz]*xsref.totXs1d(zid, ie) +   //xsref(ie, zid, 0, 0) +  //xs->operator()(ie, zid, 0, 0) +
                                    mesh->Ayz[iang*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz] +  // The 2x is already factored in
                                    mesh->Axz[iang*mesh->xElemCt*mesh->zElemCt + ix*mesh->zElemCt + iz] +
                                    mesh->Axy[iang*mesh->xElemCt*mesh->yElemCt + ix*mesh->yElemCt + iy];

                            float angFlux = numer/denom;
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
                        }
                    }
                }
                //}  // End of octant 1

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
            }

            errList.push_back(maxDiff);

            for(unsigned int i = 0; i < tempFlux.size(); i++)
            {
                //int indx = ie*m_mesh->voxelCount() + i; // TODO - delete
                scalarFlux[ie*m_mesh->voxelCount() + i] = tempFlux[i];
            }

            iterNum++;
        }

        emit signalNewIteration(scalarFlux);
    }

    qDebug() << "Time to complete: " << (std::clock() - startMoment)/(double)(CLOCKS_PER_SEC/1000) << " ms";

    OutWriter::writeScalarFluxMesh("./outlog.matmsh", *m_mesh, scalarFlux);

    for(unsigned int i = 0; i < errList.size(); i++)
        qDebug() << "Iteration " << i << " maxDiff: " << errList[i];

    return scalarFlux;
}
