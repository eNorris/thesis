#ifndef MESH_H
#define MESH_H

#include <vector>

#include "config.h"
#include "quadrature.h"

class Mesh
{
public:
    Mesh();
    Mesh(const Config &config, const Quadrature &quad);

    int xMesh;
    int yMesh;
    int zMesh;

    //int xIndex;
    //int yIndex;
    //int zIndex;

    std::vector<float> xIndex;
    std::vector<float> yIndex;
    std::vector<float> zIndex;

    std::vector<float> dx;
    std::vector<float> dy;
    std::vector<float> dz;

    std::vector<float> DA;  // Area of the yz plane a ray sees
    std::vector<float> DB;  // Area of the xz plane a ray sees
    std::vector<float> DC;  // Area of the xy plane a ray sees

    std::vector<float> vol;

    void load(const Config &config, const Quadrature &quad);
};

#endif // MESH_H
