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
    // these are the number of mesh elements
    xMesh = 89;
    yMesh = 99;
    zMesh = ceil(5.0 / (2.0 * config->sourceFrontGap));

    // Allocate space
    xIndex.resize(xMesh + 1);
    yIndex.resize(yMesh + 1);
    zIndex.resize(zMesh + 1);

    dx.resize(xMesh);
    dy.resize(yMesh);
    dz.resize(zMesh);

    //DA.resize(config->m * yMesh * zMesh);
    //DB.resize(config->m * xMesh * zMesh);
    //DC.resize(config->m * xMesh * yMesh);

    DA.resize(quad->angleCount() * yMesh * zMesh);
    DB.resize(quad->angleCount() * xMesh * zMesh);
    DC.resize(quad->angleCount() * xMesh * yMesh);

    vol.resize(xMesh * yMesh * zMesh);

    // The coordinates between mesh elements
    for(int i = 0; i < xMesh+1; i++)  // iterate the xMesh+1 to get the last bin
        xIndex[i] = i * (config->xLen / xMesh);

    for(int i = 0; i < yMesh+1; i++)
        yIndex[i] = i * (config->yLen / yMesh);

    for(int i = 0; i < zMesh+1; i++)
        zIndex[i] = i * (config->zLen / zMesh);

    // Calculate the segment sizes
    for(int i = 0; i < xMesh; i++)
        dx[i] = xIndex[i+1] - xIndex[i];

    for(int i = 0; i < yMesh; i++)
        dy[i] = yIndex[i+1] - yIndex[i];

    for(int i = 0; i < zMesh; i++)
        dz[i] = zIndex[i+1] - zIndex[i];

    for(int xIndx = 0; xIndx < xMesh; xIndx++)
        for(int yIndx = 0; yIndx < yMesh; yIndx++)
            for(int zIndx = 0; zIndx < zMesh; zIndx++)
                vol[xIndx * yMesh*zMesh + yIndx * zMesh + zIndx] = dx[xIndx] * dy[yIndx] * dz[zIndx];

    // Calculate the cell face area for each angle as well as volume
    for(int eIndx = 0; eIndx < config->m; eIndx++)
    {
        float vMu = abs(quad->mu[eIndx]);
        float vEta = abs(quad->eta[eIndx]);
        float vZi = abs(quad->zi[eIndx]);

        for(int yIndx = 0; yIndx < yMesh; yIndx++)
            for(int zIndx = 0; zIndx < zMesh; zIndx++)
                DA[eIndx * yMesh*zMesh + yIndx * zMesh + zIndx] = vMu * dy[yIndx] * dz[zIndx];

        for(int xIndx = 0; xIndx < xMesh; xIndx++)
            for(int zIndx = 0; zIndx < zMesh; zIndx++)
                DB[eIndx * xMesh*zMesh + xIndx * zMesh + zIndx] = vEta * dx[xIndx] * dz[zIndx];

        for(int xIndx = 0; xIndx < xMesh; xIndx++)
            for(int yIndx = 0; yIndx < yMesh; yIndx++)
                DC[eIndx * xMesh*yMesh + xIndx * yMesh + yIndx] = vZi * dx[xIndx] * dy[yIndx];
    }

    // 0 = air, 1 = water, 2 = lead/tungsten
    //std::vector<unsigned short> zoneId;
    zoneId.resize(xMesh * yMesh * zMesh, 0);  // Initialize to all zeros


    // Do all of the +1 need to be removed?
    int xLeftIndx  = (xMesh-1)/2 + 1 - round(config->colXLen/2/(config->xLen/xMesh));
    int xRightIndx = (xMesh-1)/2 + 1 + round(config->colXLen/2/(config->xLen/xMesh));
    int xLeftGapIndx = (xMesh-1)/2 + 1 - round(config->sourceLeftGap/(config->xLen/xMesh));
    int xRightGapIndx = (xMesh-1)/2 + 1 + round(config->sourceLeftGap/(config->xLen/xMesh));

    int yTopIndx = 0;
    int yBottomIndx = round(config->colYLen/(config->yLen/yMesh));
    int yTopGapIndx = round((config->colYLen/2 - config->sourceTopGap) / (config->yLen/yMesh));

    int zFrontIndx = (zMesh-1)/2 + 1 - round(config->colZLen/2/(config->zLen/zMesh));
    int zBackIndx = (zMesh-1)/2 + 1 + round(config->colZLen/2/(config->zLen/zMesh));
    int zFrontGapIndx = (zMesh-1)/2 + 1 - round(config->sourceFrontGap/(config->zLen/zMesh));
    int zBackGapIndx = (zMesh-1)/2 + 1 + round(config->sourceFrontGap/(config->zLen/zMesh));

    qDebug() << "x in " << xLeftIndx <<", " << xRightIndx << "  and out " << xLeftGapIndx << ", " << xRightGapIndx;
    qDebug() << "y in " << yTopIndx <<", " << yBottomIndx << "  and out " << yTopGapIndx << ", _";
    qDebug() << "z in " << zFrontIndx <<", " << zBackIndx << "  and out " << zFrontGapIndx << ", " << zBackGapIndx;


    for(int i = 0; i < xMesh; i++)
        for(int j = 0; j < yMesh; j++)
            for(int k = 0; k < zMesh; k++)
                if(insideBox(i,j,k, xLeftIndx, xRightIndx, yTopIndx, yBottomIndx, zFrontIndx, zBackIndx) &&
                        !insideBox(i,j,k, xLeftGapIndx, xRightGapIndx, yTopGapIndx, 1000000, zFrontGapIndx, zBackGapIndx))
                {
                    //qDebug() << "Setting to zone 2";
                    zoneId[i*yMesh*zMesh + j*zMesh + k] = 2;
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
    for(int i = 0; i < xMesh; i++)
        for(int j = 0; j < yMesh; j++)
            for(int k = 0; k < zMesh; k++)
            {
                float x = xIndex[i] + dx[i]/2.0;
                float y = yIndex[j] + dy[j]/2.0;
                if((x-xCenter)*(x-xCenter) + (y-yCenter)*(y-yCenter) <= (radius)*(radius))
                    zoneId[i*yMesh*zMesh + j*zMesh + k] = 1;
            }

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

int Mesh::voxelCount() const
{
    return xMesh * yMesh * zMesh;
}

int Mesh::xjmp() const
{
    return yMesh * zMesh;
}

int Mesh::yjmp() const
{
    return zMesh;
}














