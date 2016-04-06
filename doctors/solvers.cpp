//#include "solvers.h"
#include "mainwindow.h"

#include <QDebug>
#include <ctime>

#include "quadrature.h"
#include "mesh.h"
#include "xsection.h"
#include "config.h"
#include "outwriter.h"

#include "outputdialog.h"
/*
std::vector<float> MainWindow::gssolver(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const Config *config)
{

    std::clock_t startMoment = std::clock();

    std::vector<float> scalarFlux;
    std::vector<float> angularFlux;
    std::vector<float> tempFlux;
    std::vector<float> preFlux;
    std::vector<float> totalSource;
    std::vector<float> isocaSource;
    std::vector<float> errList;

    std::vector<float> extSource;

    float fluxi = 0;
    float fluxi_pre = 0;
    std::vector<float> fluxj;
    std::vector<float> fluxk;
    std::vector<float> fluxj_pre;
    std::vector<float> fluxk_pre;

    int ejmp = mesh->voxelCount() * quad->angleCount();
    int ajmp = mesh->voxelCount();
    int xjmp = mesh->xjmp();
    int yjmp = mesh->yjmp();

    scalarFlux.resize(xs->groupCount() * mesh->voxelCount());
    tempFlux.resize(mesh->voxelCount());
    angularFlux.resize(xs->groupCount() * quad->angleCount() * mesh->voxelCount());
    totalSource.resize(mesh->voxelCount());
    isocaSource.resize(mesh->voxelCount());

    extSource.resize(mesh->voxelCount(), 0.0f);

    extSource[((mesh->xMesh + 1)/2)*xjmp + ((mesh->yMesh-1)/2)*yjmp + ((mesh->zMesh+1)/2)] = 1E6;

    fluxj.resize(mesh->xMesh, 0.0f);
    fluxk.resize(mesh->xMesh * mesh->yMesh, 0.0f);
    fluxj_pre.resize(mesh->xMesh, 0.0f);
    fluxk_pre.resize(mesh->xMesh * mesh->yMesh, 0.0f);



    qDebug() << "Solving " << mesh->voxelCount() * quad->angleCount() * xs->groupCount() << " elements in phase space";

    for(int ie = 0; ie < xs->groupCount(); ie++)
    {
        qDebug() << "Energy group #" << ie;
        // Do the solving...

        int iterNum = 1;
        float maxDiff = 1.0;

        while(iterNum <= config->maxit && maxDiff > config->epsi)
        {
            qDebug() << "Iteration #" << iterNum;

            preFlux = tempFlux;  // Store flux for previous iteration

            // Calculate the scattering source
            // TODO
            for(int iie = 0; iie < ie; iie++)
                for(int iik = 0; iik < mesh->zMesh; iik++)
                    for(int iij = 0; iij < mesh->yMesh; iij++)
                        for(int iii = 0; iii < mesh->xMesh; iii++)
                        {
                            int indx = iii*mesh->xjmp() + iij*mesh->yjmp() + iik;
                            int zidIndx = mesh->zoneId[indx];
                            isocaSource[indx] = isocaSource[indx] + 1.0/(4.0*M_PI)*(scalarFlux[iie*mesh->voxelCount() + indx] * (*xs)(ie-1, zidIndx, 0, iie));
                        }

            // Calculate the total flux
            for(int i = 0; i < mesh->voxelCount(); i++)
                totalSource[i] = extSource[ie*mesh->voxelCount() + i] + isocaSource[i];

            for(unsigned int i = 0; i < tempFlux.size(); i++) // Clear for a new sweep
                tempFlux[i] = 0;

            for(int iang = 0; iang < quad->angleCount(); iang++)
            {
                qDebug() << "Angle #" << iang;

                if(quad->mu[iang] < 0 && quad->eta[iang] < 0 && quad->zi[iang] < 0)  // Octant 1
                {
                    for(int iz = mesh->zMesh-1; iz >= 0; iz--)  // Start at corner and sweep toward 0
                    {
                        for(int iy = mesh->yMesh-1; iy >= 0; iy--)
                        {
                            for(int ix = mesh->xMesh-1; ix >= 0; ix--)
                            {
                                int zid = mesh->zoneId[ix*mesh->xjmp() + iy*mesh->yjmp() + iz];

                                if(ix == mesh->xMesh-1)
                                {
                                    fluxi = 0;
                                    fluxi_pre = 0;
                                }
                                else
                                {
                                    fluxi = 2 * angularFlux[ie*ejmp + iang*ajmp + (ix+1)*xjmp + iy*yjmp + iz] - fluxi_pre;
                                    if(fluxi < 0)
                                    {
                                        fluxi = 0;
                                        // TODO - why do we add 1 instead of subtracting?
                                        angularFlux[ie*ejmp + iang*ajmp + (ix+1)*xjmp + iy*yjmp + iz] = 0.5 * (fluxi + fluxi_pre);  // TODO simplify? just set fluxi = 0
                                    }
                                }

                                if(iy == mesh->yMesh-1)
                                {
                                    fluxj[ix] = 0;
                                    fluxj_pre[ix] = 0;
                                }
                                else
                                {
                                    fluxj[ix] = 2*angularFlux[ie*ejmp + iang*ajmp + ix*xjmp + (iy+1)*yjmp + iz] - fluxj_pre[ix];
                                    if(fluxj[ix] < 0)
                                    {
                                        fluxj[ix] = 0;
                                        angularFlux[ie*ejmp + iang*ajmp + ix*xjmp + (iy+1)*yjmp + iz] = 0.5 * (fluxj[ix] + fluxj_pre[ix]);
                                    }
                                    fluxj_pre[ix] = fluxj[ix];
                                }

                                if(iz == mesh->zMesh-1)
                                {
                                    fluxk[ix*mesh->xMesh + iy] = 0;
                                    fluxk_pre[ix*mesh->xMesh + iy] = 0;
                                }
                                else
                                {
                                    fluxk[ix*mesh->xMesh + iy] = 2*angularFlux[ie*ejmp + iang*ajmp + ix*xjmp + iy*yjmp + iz + 1] - fluxk_pre[ix*mesh->xMesh + iy];
                                    if(fluxk[ix*mesh->xMesh + iy] < 0)
                                    {
                                        fluxk[ix*mesh->xMesh + iy] = 0;
                                        angularFlux[ie*ejmp + iang*ajmp + ix*xjmp + iy*yjmp + iz + 1] = 0.5 * (fluxk[ix*mesh->xMesh + iy] + fluxk_pre[ix*mesh->xMesh + iy]);
                                    }
                                    fluxk_pre[ix*mesh->xMesh + iy] = fluxk[ix*mesh->xMesh + iy];
                                }

                                float numer = totalSource[ix*xjmp+iy*yjmp+iz] * mesh->vol[ix*xjmp+iy*yjmp+iz] +
                                        2*mesh->DA[iang*mesh->yMesh*mesh->zMesh + iy*mesh->zMesh + iz] * fluxi +
                                        2*mesh->DB[iang*mesh->xMesh*mesh->zMesh + ix*mesh->zMesh + iz] * fluxj[ix] +
                                        2*mesh->DC[iang*mesh->xMesh*mesh->yMesh + ix*mesh->yMesh + iy] * fluxk[ix*mesh->xMesh + iy];
                                float denom = mesh->vol[ix*xjmp+iy*yjmp+iz]*xs->operator ()(ie, zid, 0, 0) + //xs->msig[ie*xs->dim1()*xs->dim2()*xs->dim3() + zid*xs->dim2()*xs->dim3() + xs->dim3()] +
                                        2*mesh->DA[iang*mesh->yMesh*mesh->zMesh + iy*mesh->zMesh + iz] +
                                        2*mesh->DB[iang*mesh->xMesh*mesh->zMesh + ix*mesh->zMesh + iz] +
                                        2*mesh->DC[iang*mesh->xMesh*mesh->yMesh + ix*mesh->yMesh + iy];
                                angularFlux[ie*ejmp + iang*ajmp + ix*xjmp + iy*yjmp + iz] = numer/denom;

                                //if(denom == 0)
                                //    qDebug() << "All life is over!";

                                //if(numer != 0)
                                //    qDebug() << "Got a fish!";

                                // Sum all the angular fluxes
                                tempFlux[ix*xjmp + iy*yjmp + iz] = tempFlux[ix*xjmp + iy*yjmp + iz] + quad->wt[iang]*angularFlux[ie*ejmp + iang*ajmp + ix*xjmp + iy*yjmp + iz];
                            }
                        }
                    }
                }  // End of octant 1

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
                scalarFlux[ie*ejmp + i] = tempFlux[i];

            iterNum++;
        }

        emit signalNewIteration(scalarFlux);
    }

    qDebug() << "Time to complete: " << (std::clock() - startMoment)/(double)(CLOCKS_PER_SEC/1000) << " ms";

    for(unsigned int i = 0; i < errList.size(); i++)
        qDebug() << i << " maxDiff: " << errList[i];

    return scalarFlux;
}
*/

std::vector<float> MainWindow::gssolver(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const Config *config)
{

    std::clock_t startMoment = std::clock();

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

    //extSource[((mesh->xMesh - 1)/2)*xjmp + ((mesh->yMesh-1)/2)*yjmp + ((mesh->zMesh-1)/2)] = 1E6;  // [gammas / sec]
    extSource[((mesh->xMesh - 1)/2)*xjmp + (config->colYLen/2)*yjmp + ((mesh->zMesh-1)/2)] = 1E6;

    qDebug() << "Solving " << mesh->voxelCount() * quad->angleCount() * xs->groupCount() << " elements in phase space";

    for(int ie = 0; ie < xs->groupCount(); ie++)
    {
        qDebug() << "Energy group #" << ie;
        // Do the solving...

        int iterNum = 1;
        float maxDiff = 1.0;

        while(iterNum <= config->maxit && maxDiff > config->epsi)
        {
            qDebug() << "Iteration #" << iterNum;

            preFlux = tempFlux;  // Store flux for previous iteration

            // TODO Zero the isocasource
            for(int i = 0; i < isocaSource.size(); i++)
                isocaSource[i] = 0;

            // Calculate the scattering source
            for(int iie = 0; iie <= ie; iie++)
                for(int iik = 0; iik < mesh->zMesh; iik++)
                    for(int iij = 0; iij < mesh->yMesh; iij++)
                        for(int iii = 0; iii < mesh->xMesh; iii++)
                        {
                            int indx = iii*xjmp + iij*yjmp + iik;
                            int zidIndx = mesh->zoneId[indx];
                            //float xsval = xsref.scatXs(zidIndx, iie, ie);
                            //qDebug() << xsval;
                            isocaSource[indx] += 1.0/(4.0*M_PI)*scalarFlux[iie*mesh->voxelCount() + indx] * xsref.scatXs(zidIndx, iie, ie) * mesh->vol[indx]; //xsref(ie-1, zidIndx, 0, iie));
                        }



            float isomax = -1;
            for(int ti = 0; ti < isocaSource.size(); ti++)
                if(isocaSource[ti] > isomax)
                    isomax = isocaSource[ti];

            float scamax = -1;
            for(int ti = 0; ti < scalarFlux.size(); ti++)
                if(scalarFlux[ti] > scamax)
                    scamax = scalarFlux[ti];

            // Calculate the total source
            for(int i = 0; i < mesh->voxelCount(); i++)
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

                //for(int iz = 0; iz < mesh->zMesh; iz++)  // Start at origin and sweep towards corner
                for(int iz = quad->eta[iang] >= 0 ? 0 : mesh->zMesh-1; quad->eta[iang] >= 0 ? iz < mesh->zMesh : iz >= 0; iz += (quad->eta[iang] >= 0 ? 1 : -1))
                {
                    //for(int iy = 0; iy < mesh->yMesh; iy++)
                    for(int iy = quad->zi[iang] >= 0 ? 0 : mesh->yMesh-1; quad->zi[iang] >= 0 ? iy < mesh->yMesh : iy >= 0; iy += (quad->zi[iang] >= 0 ? 1 : -1))
                    {
                        //for(int ix = 0; ix < mesh->xMesh; ix++)
                        for(int ix = quad->mu[iang] >= 0 ? 0 : mesh->xMesh-1; quad->mu[iang] >= 0 ? ix < mesh->xMesh : ix >= 0; ix += (quad->mu[iang] >= 0 ? 1 : -1))
                        {
                            int zid = mesh->zoneId[ix*xjmp + iy*yjmp + iz];

                            // Handle the x influx
                            float t1 = quad->mu[iang];
                            float t2 = quad->zi[iang];
                            float t3 = quad->eta[iang];
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
                                if(ix == mesh->xMesh-1)
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
                                if(iy == mesh->yMesh-1)
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
                                if(iz == mesh->zMesh-1)
                                    influxZ = 0.0f;
                                else
                                    influxZ = outboundFluxZ[ix*xjmp + iy*yjmp + iz+1];
                                    //influxZ = angularFlux[ie*ejmp + iang*ajmp + ix*xjmp + iy*yjmp + iz+1];
                            }

                            // I don't think the *vol should be here
                            // I'm pretty sure totalSource isn't normalized by volume...
                            float numer = totalSource[ix*xjmp+iy*yjmp+iz] + //* mesh->vol[ix*xjmp+iy*yjmp+iz] +
                                    mesh->Ayz[iang*mesh->yMesh*mesh->zMesh + iy*mesh->zMesh + iz] * influxX +  // The 2x is already factored in
                                    mesh->Axz[iang*mesh->xMesh*mesh->zMesh + ix*mesh->zMesh + iz] * influxY +
                                    mesh->Axy[iang*mesh->xMesh*mesh->yMesh + ix*mesh->yMesh + iy] * influxZ;
                            float denom = mesh->vol[ix*xjmp+iy*yjmp+iz]*xsref.totXs(zid, ie) +   //xsref(ie, zid, 0, 0) +  //xs->operator()(ie, zid, 0, 0) +
                                    mesh->Ayz[iang*mesh->yMesh*mesh->zMesh + iy*mesh->zMesh + iz] +  // The 2x is already factored in
                                    mesh->Axz[iang*mesh->xMesh*mesh->zMesh + ix*mesh->zMesh + iz] +
                                    mesh->Axy[iang*mesh->xMesh*mesh->yMesh + ix*mesh->yMesh + iy];

                            float angFlux = numer/denom;
                            angularFlux[ie*ejmp + iang*ajmp + ix*xjmp + iy*yjmp + iz] = angFlux;

                            //if(angFlux != 0)
                            //    qDebug() << "got it";
                            //if(influxX > 0 || influxY > 0 || influxZ > 0)
                            //    qDebug() << "Gotcha!";

                            outboundFluxX[ix*xjmp + iy*yjmp + iz] = 2*angFlux - influxX;
                            outboundFluxY[ix*xjmp + iy*yjmp + iz] = 2*angFlux - influxY;
                            outboundFluxZ[ix*xjmp + iy*yjmp + iz] = 2*angFlux - influxZ;

                            float outX = 2*angFlux - influxX;
                            float outY = 2*angFlux - influxY;
                            float outZ = 2*angFlux - influxZ;


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
                scalarFlux[ie*ejmp + i] = tempFlux[i];

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


/*
 * 55/56 need to be inside the energy loop
 * */
