//#include "solvers.h"
#include "mainwindow.h"

#include <QDebug>
#include <ctime>

#include "quadrature.h"
#include "mesh.h"
#include "xsection.h"
#include "config.h"

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

                emit signalDebugHalt(tempFlux);

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


/*
 * 55/56 need to be inside the energy loop
 * */
