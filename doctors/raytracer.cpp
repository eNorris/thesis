
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

    const unsigned short DIRECTION_NONE = 0;
    const unsigned short DIRECTION_X = 1;
    const unsigned short DIRECTION_Y = 2;
    const unsigned short DIRECTION_Z = 3;

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

                    if(xIndxStart == srcIndxX && yIndxStart == srcIndxY && zIndxStart == srcIndxZ)
                    {
                        // TODO: Calculate
                        float srcToCellDist = sqrt((x-sx)*(x-sx) + (y-sy)*(y-sy) + (z-sz)*(z-sz));
                        unsigned int zid = mesh->zoneId[xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart];
                        float xsval = xs->xsTot[zid];
                        for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                            uflux[ie*ejmp + xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart] = srcStrength * exp(-xsval*srcToCellDist) / (4 * M_PI * srcToCellDist * srcToCellDist);
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

                    float xcos = srcToCellX/srcToCellDist;
                    float ycos = srcToCellY/srcToCellDist;
                    float zcos = srcToCellZ/srcToCellDist;

                    int xBoundIndx = (xcos >= 0 ? xIndx+1 : xIndx);
                    int yBoundIndx = (ycos >= 0 ? yIndx+1 : yIndx);
                    int zBoundIndx = (zcos >= 0 ? zIndx+1 : zIndx);

                    for(unsigned int i = 0; i < xs->groupCount(); i++)
                        meanFreePaths[i] = 0.0f;

                    //int mchk = 1;
                    bool exhaustedRay = false;
                    while(!exhaustedRay)
                    {
                        // Determine the distance to cell boundaries
                        //float zz = fabs(xcos);
                        float tx = (fabs(xcos) < tiny ? huge : (mesh->xNodes[xBoundIndx] - x)/xcos);
                        float ty = (fabs(ycos) < tiny ? huge : (mesh->yNodes[yBoundIndx] - y)/ycos);
                        float tz = (fabs(zcos) < tiny ? huge : (mesh->zNodes[zBoundIndx] - z)/zcos);

                        // Determine the shortest distance in 3 directions
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

                        if(tmin < 0)
                            qDebug() << "Reversed space!";

                        // Calculate distance from cell to source
                        /*
                        srcToCellX = x - sx;
                        srcToCellY = y - sy;
                        srcToCellZ = z - sz;

                        float newdist = sqrt(srcToCellX*srcToCellX + srcToCellY*srcToCellY + srcToCellZ*srcToCellZ);

                        if(newdist < tmin)
                        {
                            tmin = newdist;
                            //mchk = 0;
                            exhaustedRay = true;
                            dirHitFirst = DIRECTION_NONE;
                        }
                        */

                        // Update mpf array
                        unsigned int zid = mesh->zoneId[xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart];
                        for(unsigned int ie = 0; ie < xs->groupCount(); ie++)
                        {
                            meanFreePaths[ie] += tmin * xs->xsTot[zid];
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
                            if(xIndx < 0 || xIndx >= (signed) mesh->xElemCt)
                                exhaustedRay = true;
                                //mchk = 0;
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
                            if(yIndx < 0 || yIndx >= (signed) mesh->yElemCt)
                                exhaustedRay = true;
                                //mchk = 0;
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
                            if(zIndx < 0 || zIndx >= (signed) mesh->zElemCt)
                                exhaustedRay = true;
                                //mchk = 0;
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

                        float mfp = meanFreePaths[ie];
                        float flx = srcStrength * exp(-meanFreePaths[ie]) / (4 * M_PI * srcToCellDist * srcToCellDist);

                        if(flx < 0)
                            qDebug() << "raytracer.cpp: (223): Negative?";

                        if(flx > 1E6)
                            qDebug() << "raytracer.cpp: (226): Too big!";

                        uflux[ie*ejmp + xIndxStart*xjmp + yIndxStart*yjmp + zIndxStart] = srcStrength * exp(-meanFreePaths[ie]) / (4 * M_PI * srcToCellDist * srcToCellDist);
                    }

                } // End of each voxel
    }

    qDebug() << "Time to complete raytracer: " << (std::clock() - startMoment)/(double)(CLOCKS_PER_SEC/1000) << " ms";

    emit signalNewIteration(uflux);
    return uflux;
}








