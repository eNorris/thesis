#include "mesh.h"

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

    //DA.resize(config->m * yMesh * zMesh);
    //DB.resize(config->m * xMesh * zMesh);
    //DC.resize(config->m * xMesh * yMesh);

    //DA.resize(quad->angleCount() * yElemCt * zElemCt);
    //DB.resize(quad->angleCount() * xElemCt * zElemCt);
    //DC.resize(quad->angleCount() * xElemCt * yElemCt);

    Axy.resize(quad->angleCount() * xElemCt * yElemCt);
    Ayz.resize(quad->angleCount() * yElemCt * zElemCt);
    Axz.resize(quad->angleCount() * xElemCt * zElemCt);

    /*
                                                       // mu xi eta
    orderOctant1.resize(xElemCt * yElemCt * zElemCt);  // + + +
    orderOctant2.resize(xElemCt * yElemCt * zElemCt);  // - + +
    orderOctant3.resize(xElemCt * yElemCt * zElemCt);  // - - +
    orderOctant4.resize(xElemCt * yElemCt * zElemCt);  // + - +
    orderOctant5.resize(xElemCt * yElemCt * zElemCt);  // + + -
    orderOctant6.resize(xElemCt * yElemCt * zElemCt);  // - + -
    orderOctant7.resize(xElemCt * yElemCt * zElemCt);  // - - -
    orderOctant8.resize(xElemCt * yElemCt * zElemCt);  // + - -

    unsigned int indx = 0;  // Octant 1: + + +
    for(int i = 0; i < xElemCt; i++)
        for(int j = 0; j < yElemCt; j++)
            for(int k = 0; k < zElemCt; k++)
                orderOctant1[indx++] = i*yElemCt*zElemCt + j*zElemCt + k;

    indx = 0;  // Octant 2: - + +
    for(int i = xElemCt-1; i >= 0; i--)
        for(int j = 0; j < yElemCt; j++)
            for(int k = 0; k < zElemCt; k++)
                orderOctant1[indx++] = i*yElemCt*zElemCt + j*zElemCt + k;

    indx = 0;  // Octant 3: - - +
    for(int i = xElemCt-1; i >= 0; i--)
        for(int j = yElemCt-1; j >= 0; j--)
            for(int k = 0; k < zElemCt; k++)
                orderOctant1[indx++] = i*yElemCt*zElemCt + j*zElemCt + k;

    indx = 0;  // Octant 4: + - +
    for(int i = 0; i < xElemCt; i++)
        for(int j = yElemCt-1; j >= 0; j--)
            for(int k = 0; k < zElemCt; k++)
                orderOctant1[indx++] = i*yElemCt*zElemCt + j*zElemCt + k;

    indx = 0;  // Octant 5: + + -
    for(int i = 0; i < xElemCt; i++)
        for(int j = 0; j < yElemCt; j++)
            for(int k = zElemCt-1; k >= 0; k--)
                orderOctant1[indx++] = i*yElemCt*zElemCt + j*zElemCt + k;

    indx = 0;  // Octant 6: - + -
    for(int i = xElemCt-1; i >= 0; i--)
        for(int j = 0; j < yElemCt; j++)
            for(int k = zElemCt-1; k >= 0; k--)
                orderOctant1[indx++] = i*yElemCt*zElemCt + j*zElemCt + k;

    indx = 0;  // Octant 7: - - -
    for(int i = xElemCt-1; i >= 0; i--)
        for(int j = yElemCt-1; j >= 0; j--)
            for(int k = zElemCt-1; k >= 0; k--)
                orderOctant1[indx++] = i*yElemCt*zElemCt + j*zElemCt + k;

    indx = 0;  // Octant 8: + - -
    for(int i = 0; i < zElemCt; i++)
        for(int j = yElemCt-1; j >= 0; j--)
            for(int k = zElemCt-1; k >= 0; k--)
                orderOctant1[indx++] = i*yElemCt*zElemCt + j*zElemCt + k;
    */


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

    // Calculate the cell face area for each angle as well as volume
    for(int eIndx = 0; eIndx < config->m; eIndx++)
    {
        float vMu = fabs(quad->mu[eIndx]);
        float vEta = fabs(quad->eta[eIndx]);
        float vZi = fabs(quad->zi[eIndx]);

        for(unsigned int yIndx = 0; yIndx < yElemCt; yIndx++)
            for(unsigned int zIndx = 0; zIndx < zElemCt; zIndx++)
            {
                //DA[eIndx * yElemCt*zElemCt + yIndx * zElemCt + zIndx] = vMu * dy[yIndx] * dz[zIndx];
                Ayz[eIndx * yElemCt*zElemCt + yIndx * zElemCt + zIndx] = 2 * vMu * dy[yIndx] * dz[zIndx];
            }

        for(unsigned int xIndx = 0; xIndx < xElemCt; xIndx++)
            for(unsigned int zIndx = 0; zIndx < zElemCt; zIndx++)
            {
                //DB[eIndx * xElemCt*zElemCt + xIndx * zElemCt + zIndx] = vEta * dx[xIndx] * dz[zIndx];
                Axz[eIndx * xElemCt*zElemCt + xIndx * zElemCt + zIndx] = 2 * vZi * dx[xIndx] * dz[zIndx];
            }

        for(unsigned int xIndx = 0; xIndx < xElemCt; xIndx++)
            for(unsigned int yIndx = 0; yIndx < yElemCt; yIndx++)
            {
                //DC[eIndx * xElemCt*yElemCt + xIndx * yElemCt + yIndx] = vZi * dx[xIndx] * dy[yIndx];
                Axy[eIndx * xElemCt*yElemCt + xIndx * yElemCt + yIndx] = 2 * vEta * dx[xIndx] * dy[yIndx];
            }
    }

    // 0 = air, 1 = water, 2 = lead/tungsten
    //std::vector<unsigned short> zoneId;
    zoneId.resize(xElemCt * yElemCt * zElemCt, 0);  // Initialize to all zeros


    // Do all of the +1 need to be removed?
    int xLeftIndx  = (xElemCt-1)/2 + 1 - round(config->colXLen/2/(config->xLen/xElemCt));
    int xRightIndx = (xElemCt-1)/2 + 1 + round(config->colXLen/2/(config->xLen/xElemCt));
    int xLeftGapIndx = (xElemCt-1)/2 + 1 - round(config->sourceLeftGap/(config->xLen/xElemCt));
    int xRightGapIndx = (xElemCt-1)/2 + 1 + round(config->sourceLeftGap/(config->xLen/xElemCt));

    int yTopIndx = 0;
    int yBottomIndx = round(config->colYLen/(config->yLen/yElemCt));
    int yTopGapIndx = round((config->colYLen/2 - config->sourceTopGap) / (config->yLen/yElemCt));

    int zFrontIndx = (zElemCt-1)/2 + 1 - round(config->colZLen/2/(config->zLen/zElemCt));
    int zBackIndx = (zElemCt-1)/2 + 1 + round(config->colZLen/2/(config->zLen/zElemCt));
    int zFrontGapIndx = (zElemCt-1)/2 + 1 - round(config->sourceFrontGap/(config->zLen/zElemCt));
    int zBackGapIndx = (zElemCt-1)/2 + 1 + round(config->sourceFrontGap/(config->zLen/zElemCt));

    qDebug() << "x in " << xLeftIndx <<", " << xRightIndx << "  and out " << xLeftGapIndx << ", " << xRightGapIndx;
    qDebug() << "y in " << yTopIndx <<", " << yBottomIndx << "  and out " << yTopGapIndx << ", _";
    qDebug() << "z in " << zFrontIndx <<", " << zBackIndx << "  and out " << zFrontGapIndx << ", " << zBackGapIndx;


    for(unsigned int i = 0; i < xElemCt; i++)
        for(unsigned int j = 0; j < yElemCt; j++)
            for(unsigned int k = 0; k < zElemCt; k++)
                if(insideBox(i,j,k, xLeftIndx, xRightIndx, yTopIndx, yBottomIndx, zFrontIndx, zBackIndx) &&
                        !insideBox(i,j,k, xLeftGapIndx, xRightGapIndx, yTopGapIndx, 1000000, zFrontGapIndx, zBackGapIndx))
                {
                    //qDebug() << "Setting to zone 2";
                    zoneId[i*yElemCt*zElemCt + j*zElemCt + k] = 2;
                }
                // Not sure if these should be < or <=
                //if(xLeftIndx <= i && i <= xRightIndx &&  // If inside collimator
                //   yTopIndx <= j && j <= yBottomIndx &&
                //   zFrontIndx <= k && k <= zBackIndx  &&
                //   (xLeftGapIndx >= i && i >= xRightGapIndx &&  // and outside gap
                //   yTopGapIndx >= j &&  // Not bottom gap
                //   zFrontGapIndx >= k && k >= zBackGapIndx))
                //{
                //    qDebug() << "Setting to zone 2";
                //    zoneId[i*yMesh*zMesh + j*zMesh + k] = 2;
                //}


    //left_x = (cfg.xmesh-1)/2+1-round(cfg.col_xlen/2/(cfg.xlen/cfg.xmesh));             % collimator x left index
    //right_x = (cfg.xmesh-1)/2+1+round(cfg.col_xlen/2/(cfg.xlen/cfg.xmesh));            % collimator x right index
    //left_gap_x = (cfg.xmesh-1)/2+1-round(cfg.source_left_gap/(cfg.xlen/cfg.xmesh));    % collimator x gap left index
    //right_gap_x = (cfg.xmesh-1)/2+1+round(cfg.source_right_gap/(cfg.xlen/cfg.xmesh));  % collimator x gap right index

    //top_y = 1;                                                                         % collimator y direction
    //bottom_y = round(cfg.col_ylen/(cfg.ylen/cfg.ymesh));
    //top_gap_y = round((cfg.col_ylen/2-cfg.source_top_gap)/(cfg.ylen/cfg.ymesh));       % the source is at the center of the collimator

    //front_z = (cfg.zmesh-1)/2+1-round(cfg.col_zlen/2/(cfg.zlen/cfg.zmesh));            % collimator z front index
    //back_z = (cfg.zmesh-1)/2+1+round(cfg.col_zlen/2/(cfg.zlen/cfg.zmesh));             % collimator x right index
    //front_gap_z = (cfg.zmesh-1)/2+1-round(cfg.source_front_gap/(cfg.zlen/cfg.zmesh));  % collimator x gap left index
    //back_gap_z = (cfg.zmesh-1)/2+1+round(cfg.source_back_gap/(cfg.zlen/cfg.zmesh));    % collimator x gap right index

    //zone_id(left_x:right_x, top_y:bottom_y, front_z:back_z) = 2;                       % set collimator zone as 2

    //zone_id(left_gap_x:right_gap_x, top_gap_y:bottom_y, front_gap_z:back_gap_z) = 0;   % set the hole in collimator zone 0

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










