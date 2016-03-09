#ifndef MESH_H
#define MESH_H

#include <vector>

#include "config.h"
#include "quadrature.h"

class Mesh
{
public:
    Mesh();

    int xMesh;
    int yMesh;
    int zMesh;

    int xIndex;
    int yIndex;
    int zIndex;

    void load(const Config &config, const Quadrature &quad);
};

#endif // MESH_H
