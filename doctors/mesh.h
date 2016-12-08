#ifndef MESH_H
#define MESH_H

#include <vector>

#include <QObject>

#include "config.h"
#include "quadrature.h"

class Mesh : public QObject
{
    Q_OBJECT
public:
    Mesh();
    Mesh(const Config *config, const Quadrature *quad);

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

    //std::vector<float> DA;  // Area of the yz plane a ray sees
    //std::vector<float> DB;  // Area of the xz plane a ray sees
    //std::vector<float> DC;  // Area of the xy plane a ray sees

    std::vector<float> Axy;  // Replaces DA (Alreadys has the x2 factored in
    std::vector<float> Ayz;  // Replaces DB
    std::vector<float> Axz;  // Replaces DC

    /*
                                             // mu xi eta
    std::vector<unsigned int> orderOctant1;  // + + +
    std::vector<unsigned int> orderOctant2;  // - + +
    std::vector<unsigned int> orderOctant3;  // - - +
    std::vector<unsigned int> orderOctant4;  // + - +
    std::vector<unsigned int> orderOctant5;  // + + -
    std::vector<unsigned int> orderOctant6;  // - + -
    std::vector<unsigned int> orderOctant7;  // - - -
    std::vector<unsigned int> orderOctant8;  // + - -
    */

    // Always initialized
    std::vector<unsigned short> zoneId;
    std::vector<float> vol;

    // Only initialized when reading CT data
    std::vector<u_int16_t> ct;

    /** density in [g/cm^3] */
    std::vector<float> density;

    /** atom density in [atom/b-cm] */
    std::vector<float> atomDensity;

    void load(const Config *config, const Quadrature *quad);

    unsigned int voxelCount() const;
    int xjmp() const;
    int yjmp() const;

    //std::vector<unsigned int> &getOctantOrder(const float mu, const float xi, const float eta);

public slots:
    void remesh(int xelems, int yelems, int zelems, const Config *config, const Quadrature *quad);
    void uniform(const int xelems, const int yelems, const int zelems, const float xLen, const float yLen, const float zLen);

public:
    void calcAreas(const Quadrature *quad, const int eGroups);
    void initCtVariables();

    int getFlatIndex(int xindx, int yindx, int zindx) const;
    int getZoneIdAt(int xindx, int yindx, int zindx) const;
    float getPhysicalDensityAt(int xindx, int yindx, int zindx) const;
    float getAtomDensityAt(int xindx, int yindx, int zindx) const;


private:
    bool insideBox(int x, int y, int z, int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);
    bool insideTightBox(int x, int y, int z, int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);
};

#endif // MESH_H
