
#include "mainwindow.h"

#include "quadrature.h"
#include "mesh.h"
#include "xsection.h"
#include "config.h"

#include <QDebug>

std::vector<float> MainWindow::raytrace(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const Config *config)
{
    std::vector<float> uflux;
    uflux.resize(xs->groupCount() * mesh->voxelCount());

    unsigned int ejmp = mesh->voxelCount();
    unsigned int xjmp = mesh->xjmp();
    unsigned int yjmp = mesh->yjmp();

    qDebug() << "Running raytracer";

    float tiny = 1.0E-35;
    float huge = 1.0E35;
    float e3 = 1.0E-8;
    float e4 = 2.5E-9;
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

                    unsigned int xIndx = ix;
                    unsigned int yIndx = iy;
                    unsigned int zIndx = iz;

                    // TODO: Do these need to reverse direction? Right now it's source -> cell
                    float srcToCellX = x - sx;
                    float srcToCellY = y - sy;
                    float srcToCellZ = z - sz;

                    float srcToCellDist = sqrt(srcToCellX*srcToCellX + srcToCellY*srcToCellY + srcToCellZ*srcToCellZ);

                    float xcos = srcToCellX/srcToCellDist;
                    float ycos = srcToCellY/srcToCellDist;
                    float zcos = srcToCellZ/srcToCellDist;

                    unsigned int xBoundIndx = (xcos >= 0 ? ix : ix+1);
                    unsigned int yBoundIndx = (ycos >= 0 ? iy : iy+1);
                    unsigned int zBoundIndx = (zcos >= 0 ? iz : iz+1);

                    int mchk = 1;
                    while(mchk > 0)
                    {
                        // Determine the distance to cell boundaries
                        float tx = abs(xcos) < tiny ? huge : (mesh->xNodes[xBoundIndx] - x)/xcos;
                        float ty = abs(ycos) < tiny ? huge : (mesh->yNodes[yBoundIndx] - y)/ycos;
                        float tz = abs(zcos) < tiny ? huge : (mesh->zNodes[zBoundIndx] - z)/zcos;

                        // Determine the shortest distance in 3 directions
                        float tmin;
                        float dirIndx;

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
                        srcToCellX = x - sx;
                        srcToCellY = y - sy;
                        srcToCellZ = z - sz;

                        float newdist = sqrt(srcToCellX*srcToCellX + srcToCellY*srcToCellY + srcToCellZ*srcToCellZ);

                        if(newdist < srcToCellDist)
                        {
                            tmin = newdist;
                            mchk = 0;
                            dirIndx = 0;
                        }

                        // Update mpf array
                        int zid = mesh->zoneId[ix*xjmp + iy*yjmp + iz];
                        for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
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
                            if(xIndx < 0 || xIndx >= mesh->xElemCt)
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

                    for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                    {
                        // TODO - For now all sources emit the same strength for all energies
                        uflux[ie*ejmp + ix*xjmp + iy*yjmp + iz] = config->sourceIntensity[is] * exp(-meanFreePaths[ie]) / (4 * M_PI * srcToCellDist * srcToCellDist);
                    }

                } // End of each voxel
    }

    emit signalNewIteration(uflux);
    return uflux;
}








