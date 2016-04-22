
#include "mainwindow.h"

#include "quadrature.h"
#include "mesh.h"
#include "xsection.h"
#include "config.h"

#include <ctime>

#include <QDebug>

std::vector<float> MainWindow::raytrace(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const Config *config)
{
    std::clock_t startMoment = std::clock();

    std::vector<float> uflux;
    uflux.resize(xs->groupCount() * mesh->voxelCount());

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

    for(unsigned int is = 0; is < config->sourceIntensity.size(); is++)
    {
        float sx = config->sourceX[is];
        float sy = config->sourceY[is];
        float sz = config->sourceZ[is];
        float srcStrength = config->sourceIntensity[is];

        for(unsigned int iz = 0; iz < mesh->zElemCt; iz++)
            for(unsigned int iy = 0; iy < mesh->yElemCt; iy++)
                for(unsigned int ix = 0; ix < mesh->xElemCt; ix++)
                {
                    float x = mesh->xNodes[ix] + mesh->dx[ix]/2;
                    float y = mesh->yNodes[iy] + mesh->dy[iy]/2;
                    float z = mesh->zNodes[iz] + mesh->dz[iz]/2;

                    int xIndx = ix;
                    int yIndx = iy;
                    int zIndx = iz;

                    // TODO: Do these need to reverse direction? Right now it's source -> cell
                    float srcToCellX = sx - x;
                    float srcToCellY = sy - y;
                    float srcToCellZ = sz - z;

                    float srcToCellDist = sqrt(srcToCellX*srcToCellX + srcToCellY*srcToCellY + srcToCellZ*srcToCellZ);

                    float xcos = srcToCellX/srcToCellDist;
                    float ycos = srcToCellY/srcToCellDist;
                    float zcos = srcToCellZ/srcToCellDist;

                    int xBoundIndx = (xcos >= 0 ? ix+1 : ix);
                    int yBoundIndx = (ycos >= 0 ? iy+1 : iy);
                    int zBoundIndx = (zcos >= 0 ? iz+1 : iz);

                    int mchk = 1;
                    while(mchk > 0)
                    {
                        // Determine the distance to cell boundaries
                        //float zz = fabs(xcos);
                        float tx = (fabs(xcos) < tiny ? huge : (mesh->xNodes[xBoundIndx] - x)/xcos);
                        float ty = (fabs(ycos) < tiny ? huge : (mesh->yNodes[yBoundIndx] - y)/ycos);
                        float tz = (fabs(zcos) < tiny ? huge : (mesh->zNodes[zBoundIndx] - z)/zcos);

                        // Determine the shortest distance in 3 directions
                        float tmin;
                        int dirIndx;

                        if(tx < ty && tx < tz)
                        {
                            tmin = tx;
                            dirIndx = 1;
                        }
                        else if(ty < tz)
                        {
                            tmin = ty;
                            dirIndx = 2;
                        }
                        else
                        {
                            tmin = tz;
                            dirIndx = 3;
                        }

                        // Calculate distance from cell to source
                        srcToCellX = sx - x;
                        srcToCellY = sy - y;
                        srcToCellZ = sz - z;

                        float newdist = sqrt(srcToCellX*srcToCellX + srcToCellY*srcToCellY + srcToCellZ*srcToCellZ);

                        if(newdist < tmin)
                        {
                            tmin = newdist;
                            mchk = 0;
                            dirIndx = 0;
                        }

                        // Update mpf array
                        unsigned int zid = mesh->zoneId[ix*xjmp + iy*yjmp + iz];
                        for(int ie = 0; ie < xs->groupCount(); ie++)
                            meanFreePaths[ie] += tmin * xs->xsTot[zid];

                        // Update cell indices and positions
                        if(dirIndx == 1) // x direction
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
                            if(xIndx < 0 || xIndx >= mesh->xElemCt)  // The u denotes unsigned
                                mchk = 0;
                        }
                        else if(dirIndx == 2) // y direction
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
                            if(yIndx < 0 || yIndx >= mesh->yElemCt)
                                mchk = 0;
                        }
                        else if(dirIndx == 3) // z direction
                        {
                            x += tmin*xcos;
                            y += tmin*ycos;
                            z += mesh->zNodes[zBoundIndx];
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
                            if(zIndx < 0 || zIndx >= mesh->zElemCt)
                                mchk = 0;
                        }
                    } // End of while loop

                    for(int ie = 0; ie < xs->groupCount(); ie++)
                    {
                        // TODO - For now all sources emit the same strength for all energies
                        uflux[ie*ejmp + ix*xjmp + iy*yjmp + iz] = srcStrength * exp(-meanFreePaths[ie]) / (4 * M_PI * srcToCellDist * srcToCellDist);
                    }

                } // End of each voxel
    }

    qDebug() << "Time to complete raytracer: " << (std::clock() - startMoment)/(double)(CLOCKS_PER_SEC/1000) << " ms";

    emit signalNewIteration(uflux);
    return uflux;
}








