#ifndef MESH_H
#define MESH_H

#include "globals.h"

#include <vector>

//#include <QObject>

//#include <cstdint>

#include "config.h"
#include "quadrature.h"

class Mesh
{

public:
    Mesh();
    //Mesh(const Config *config, const Quadrature *quad);

    int material;

    unsigned int xElemCt;
    unsigned int yElemCt;
    unsigned int zElemCt;

    unsigned int xNodeCt;
    unsigned int yNodeCt;
    unsigned int zNodeCt;

    std::vector<float> xNodes;
    std::vector<float> yNodes;
    std::vector<float> zNodes;

    std::vector<float> dx;
    std::vector<float> dy;
    std::vector<float> dz;

    std::vector<float> Axy;  // Replaces DA (Alreadys has the x2 factored in
    std::vector<float> Ayz;  // Replaces DB
    std::vector<float> Axz;  // Replaces DC

    // Always initialized
    //std::vector<unsigned short> zoneId;
    std::vector<int> zoneId;
    std::vector<float> vol;

    // Only initialized when reading CT data
    std::vector<U16_T> ct;

    /** density in [g/cm^3] */
    std::vector<float> density;

    /** atom density in [atom/b-cm] */
    std::vector<float> atomDensity;

    float getMaxX() const {return m_maxX;}
    float getMaxY() const {return m_maxY;}
    float getMaxZ() const {return m_maxZ;}

    //void load(const Config *config, const Quadrature *quad);

    unsigned int voxelCount() const;
    int xjmp() const;
    int yjmp() const;

public:
    void calcAreas(const Quadrature *quad, const int eGroups);
    void initCtVariables();

    int getFlatIndex(unsigned int xindx, unsigned int yindx, unsigned int zindx) const;
    int getZoneIdAt(int xindx, int yindx, int zindx) const;
    float getPhysicalDensityAt(int xindx, int yindx, int zindx) const;
    float getAtomDensityAt(int xindx, int yindx, int zindx) const;

    void remesh(int xelems, int yelems, int zelems, const Config *config, const Quadrature *quad);
    void uniform(const int xelems, const int yelems, const int zelems, const float xLen, const float yLen, const float zLen);


private:
    bool insideBox(int x, int y, int z, int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);
    bool insideTightBox(int x, int y, int z, int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);

    float m_maxX;
    float m_maxY;
    float m_maxZ;
};

#endif // MESH_H
