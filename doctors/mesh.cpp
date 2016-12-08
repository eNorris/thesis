#include "mesh.h"
#include "xsection.h"

#include <cmath>

#include <QDebug>

Mesh::Mesh()
{

}

Mesh::Mesh(const Config *config, const Quadrature *quad)
{
    load(config, quad);
}

void Mesh::load(const Config *config, const Quadrature *quad)
{
    remesh(89, 99, -1, config, quad);
    //remesh(11, 11, -1, config, quad);
}

bool Mesh::insideBox(int x, int y, int z, int xmin, int xmax, int ymin, int ymax, int zmin, int zmax)
{
    return x >= xmin && x <= xmax &&
            y >= ymin && y <= ymax &&
            z >= zmin && z <= zmax;
}

bool Mesh::insideTightBox(int x, int y, int z, int xmin, int xmax, int ymin, int ymax, int zmin, int zmax)
{
    return x > xmin && x < xmax &&
            y > ymin && y < ymax &&
            z > zmin && z < zmax;
}

unsigned int Mesh::voxelCount() const
{
    return xElemCt * yElemCt * zElemCt;
}

int Mesh::xjmp() const
{
    return yElemCt * zElemCt;
}

int Mesh::yjmp() const
{
    return zElemCt;
}

void Mesh::uniform(const int xelems, const int yelems, const int zelems, const float xLen, const float yLen, const float zLen)
{
    xElemCt = xelems;
    yElemCt = yelems;
    zElemCt = zelems;

    // Calculate the number of nodes
    xNodeCt = xElemCt + 1;
    yNodeCt = yElemCt + 1;
    zNodeCt = zElemCt + 1;

    // Allocate space
    xNodes.resize(xNodeCt);
    yNodes.resize(yNodeCt);
    zNodes.resize(zNodeCt);

    dx.resize(xElemCt);
    dy.resize(yElemCt);
    dz.resize(zElemCt);

    vol.resize(xElemCt * yElemCt * zElemCt);

    // The coordinates between mesh elements
    for(unsigned int i = 0; i < xElemCt+1; i++)  // iterate the xMesh+1 to get the last bin
        xNodes[i] = i * (xLen / xElemCt);

    for(unsigned int i = 0; i < yElemCt+1; i++)
        yNodes[i] = i * (yLen / yElemCt);

    for(unsigned int i = 0; i < zElemCt+1; i++)
        zNodes[i] = i * (zLen / zElemCt);

    // Calculate the segment sizes
    for(unsigned int i = 0; i < xElemCt; i++)
        dx[i] = xNodes[i+1] - xNodes[i];

    for(unsigned int i = 0; i < yElemCt; i++)
        dy[i] = yNodes[i+1] - yNodes[i];

    for(unsigned int i = 0; i < zElemCt; i++)
        dz[i] = zNodes[i+1] - zNodes[i];

    for(unsigned int xIndx = 0; xIndx < xElemCt; xIndx++)
        for(unsigned int yIndx = 0; yIndx < yElemCt; yIndx++)
            for(unsigned int zIndx = 0; zIndx < zElemCt; zIndx++)
                vol[xIndx * yElemCt*zElemCt + yIndx * zElemCt + zIndx] = dx[xIndx] * dy[yIndx] * dz[zIndx];

    //calcAreas(quad, eGroups);

    zoneId.resize(xElemCt * yElemCt * zElemCt, 0);  // Initialize to all zeros

}

void Mesh::remesh(int xelems, int yelems, int zelems, const Config *config, const Quadrature *quad)
{
    // these are the number of mesh elements
    xElemCt = xelems; //89;
    yElemCt = yelems; //99;
    if(zelems <= 0)
        zElemCt = ceil(5.0 / (2.0 * config->sourceFrontGap));
    else
        zElemCt = zelems;

    // Calculate the number of nodes
    xNodeCt = xElemCt + 1;
    yNodeCt = yElemCt + 1;
    zNodeCt = zElemCt + 1;

    // Allocate space
    xNodes.resize(xNodeCt);
    yNodes.resize(yNodeCt);
    zNodes.resize(zNodeCt);

    dx.resize(xElemCt);
    dy.resize(yElemCt);
    dz.resize(zElemCt);


    vol.resize(xElemCt * yElemCt * zElemCt);

    // The coordinates between mesh elements
    for(unsigned int i = 0; i < xElemCt+1; i++)  // iterate the xMesh+1 to get the last bin
        xNodes[i] = i * (config->xLen / xElemCt);

    for(unsigned int i = 0; i < yElemCt+1; i++)
        yNodes[i] = i * (config->yLen / yElemCt);

    for(unsigned int i = 0; i < zElemCt+1; i++)
        zNodes[i] = i * (config->zLen / zElemCt);

    // Calculate the segment sizes
    for(unsigned int i = 0; i < xElemCt; i++)
        dx[i] = xNodes[i+1] - xNodes[i];

    for(unsigned int i = 0; i < yElemCt; i++)
        dy[i] = yNodes[i+1] - yNodes[i];

    for(unsigned int i = 0; i < zElemCt; i++)
        dz[i] = zNodes[i+1] - zNodes[i];

    for(unsigned int xIndx = 0; xIndx < xElemCt; xIndx++)
        for(unsigned int yIndx = 0; yIndx < yElemCt; yIndx++)
            for(unsigned int zIndx = 0; zIndx < zElemCt; zIndx++)
                vol[xIndx * yElemCt*zElemCt + yIndx * zElemCt + zIndx] = dx[xIndx] * dy[yIndx] * dz[zIndx];

    calcAreas(quad, config->m);

    // 0 = air, 1 = water, 2 = lead/tungsten
    zoneId.resize(xElemCt * yElemCt * zElemCt, 0);  // Initialize to all zeros


    // Do all of the +1 need to be removed?
    int xLeftIndx  = (xElemCt-1)/2 - round(config->colXLen/2/(config->xLen/xElemCt));
    int xRightIndx = (xElemCt-1)/2 + round(config->colXLen/2/(config->xLen/xElemCt));
    int xLeftGapIndx = (xElemCt-1)/2 - round(config->sourceLeftGap/(config->xLen/xElemCt));
    int xRightGapIndx = (xElemCt-1)/2 + round(config->sourceLeftGap/(config->xLen/xElemCt));

    int yTopIndx = 0;
    int yBottomIndx = round(config->colYLen/(config->yLen/yElemCt)) - 1;
    int yTopGapIndx = round((config->colYLen/2 - config->sourceTopGap) / (config->yLen/yElemCt)) - 1;

    int zFrontIndx = (zElemCt-1)/2 + 1 - round(config->colZLen/2/(config->zLen/zElemCt));
    int zBackIndx = (zElemCt-1)/2 + round(config->colZLen/2/(config->zLen/zElemCt)) - 1;
    int zFrontGapIndx = (zElemCt-1)/2 - round(config->sourceFrontGap/(config->zLen/zElemCt));
    int zBackGapIndx = (zElemCt-1)/2 + round(config->sourceFrontGap/(config->zLen/zElemCt));

    //qDebug() << "x in " << xLeftIndx <<", " << xRightIndx << "  and out " << xLeftGapIndx << ", " << xRightGapIndx;
    //qDebug() << "y in " << yTopIndx <<", " << yBottomIndx << "  and out " << yTopGapIndx << ", _";
    //qDebug() << "z in " << zFrontIndx <<", " << zBackIndx << "  and out " << zFrontGapIndx << ", " << zBackGapIndx;

    for(unsigned int i = 0; i < xElemCt; i++)
        for(unsigned int j = 0; j < yElemCt; j++)
            for(unsigned int k = 0; k < zElemCt; k++)
                if(insideBox(i,j,k, xLeftIndx, xRightIndx, yTopIndx, yBottomIndx, zFrontIndx, zBackIndx) &&
                        !insideBox(i,j,k, xLeftGapIndx, xRightGapIndx, yTopGapIndx, 1000000, zFrontGapIndx, zBackGapIndx))
                {
                    zoneId[i*yElemCt*zElemCt + j*zElemCt + k] = 2;
                }

    float radius = 17.5;
    float xCenter = config->xLen/2.0;
    float yCenter = 50.0;  //config.yLen - config.so

    //TODO - The above yCenter should actually be calculated
    for(unsigned int i = 0; i < xElemCt; i++)
        for(unsigned int j = 0; j < yElemCt; j++)
            for(unsigned int k = 0; k < zElemCt; k++)
            {
                float x = xNodes[i] + dx[i]/2.0;
                float y = yNodes[j] + dy[j]/2.0;
                if((x-xCenter)*(x-xCenter) + (y-yCenter)*(y-yCenter) <= (radius)*(radius))
                    zoneId[i*yElemCt*zElemCt + j*zElemCt + k] = 1;
            }
}

void Mesh::calcAreas(const Quadrature *quad, const int eGroups)
{
    Axy.resize(eGroups * quad->angleCount() * xElemCt * yElemCt);
    Ayz.resize(eGroups * quad->angleCount() * yElemCt * zElemCt);
    Axz.resize(eGroups * quad->angleCount() * xElemCt * zElemCt);

    //int tmptick = 0;

    // Calculate the cell face area for each angle as well as volume
    for(int eIndx = 0; eIndx < eGroups; eIndx++)
    {
        for(int qIndx = 0; qIndx < quad->angleCount(); qIndx++)
        {
            float vMu = fabs(quad->mu[qIndx]);
            float vEta = fabs(quad->eta[qIndx]);
            float vZi = fabs(quad->zi[qIndx]);

            for(unsigned int yIndx = 0; yIndx < yElemCt; yIndx++)
                for(unsigned int zIndx = 0; zIndx < zElemCt; zIndx++)
                {
                    //DA[eIndx * yElemCt*zElemCt + yIndx * zElemCt + zIndx] = vMu * dy[yIndx] * dz[zIndx];
                    Ayz[eIndx*quad->angleCount()*yElemCt*zElemCt + qIndx*yElemCt*zElemCt + yIndx*zElemCt + zIndx] = 2 * vMu * dy[yIndx] * dz[zIndx];
                }

            for(unsigned int xIndx = 0; xIndx < xElemCt; xIndx++)
                for(unsigned int zIndx = 0; zIndx < zElemCt; zIndx++)
                {
                    //DB[eIndx * xElemCt*zElemCt + xIndx * zElemCt + zIndx] = vEta * dx[xIndx] * dz[zIndx];
                    Axz[eIndx*quad->angleCount()*xElemCt*zElemCt + qIndx*xElemCt*zElemCt + xIndx*zElemCt + zIndx] = 2 * vZi * dx[xIndx] * dz[zIndx];
                }

            for(unsigned int xIndx = 0; xIndx < xElemCt; xIndx++)
                for(unsigned int yIndx = 0; yIndx < yElemCt; yIndx++)
                {
                    Axy[eIndx*quad->angleCount()*xElemCt*yElemCt + qIndx*xElemCt*yElemCt + xIndx*yElemCt + yIndx] = 2 * vEta * dx[xIndx] * dy[yIndx];
                }
        }
    }

    //for(int chk = 0; chk < Axy.size(); chk++)
    //{
    //    if(Axy[chk] <= 1E-6)
    //        qDebug() << "EXPLODE NOW!";
    //}

}

void Mesh::initCtVariables()
{
    if(vol.size() < 1)
        throw "Cannot initialize CT variables before volume data!";
    ct.resize(vol.size());
    density.resize(vol.size());
    atomDensity.resize(vol.size());
}

int Mesh::getFlatIndex(int xindx, int yindx, int zindx) const
{
    if(xindx > xElemCt)
    {
        qDebug() << __FILE__ << ": " << __LINE__ << ": x-index was too large (" << xindx << "/" << xElemCt << ")";
    }

    if(yindx > yElemCt)
    {
        qDebug() << __FILE__ << ": " << __LINE__ << ": y-index was too large (" << yindx << "/" << yElemCt << ")";
    }

    if(zindx > zElemCt)
    {
        qDebug() << __FILE__ << ": " << __LINE__ << ": z-index was too large (" << zindx << "/" << zElemCt << ")";
    }

    return xindx * yElemCt * zElemCt + yindx * zElemCt + zindx;
}
int Mesh::getZoneIdAt(int xindx, int yindx, int zindx) const
{
    return zoneId[getFlatIndex(xindx, yindx, zindx)];
}
float Mesh::getPhysicalDensityAt(int xindx, int yindx, int zindx) const
{
    return density[getFlatIndex(xindx, yindx, zindx)];
}
float Mesh::getAtomDensityAt(int xindx, int yindx, int zindx) const
{
    return atomDensity[getFlatIndex(xindx, yindx, zindx)];
}

/*
std::vector<unsigned int> &Mesh::getOctantOrder(const float mu, const float xi, const float eta)
{
    if(eta >= 0)
    {
        if(xi >= 0)
        {
            if(mu >= 0)
                return orderOctant1;
            else
                return orderOctant2;
        }
        else
        {
            if(mu >= 0)
                return orderOctant4;
            else
                return orderOctant3;
        }
    }
    else
    {
        if(xi >= 0)
        {
            if(mu >= 0)
                return orderOctant5;
            else
                return orderOctant6;
        }
        else
        {
            if(mu >= 0)
                return orderOctant8;
            else
                return orderOctant7;
        }
    }
}
*/










