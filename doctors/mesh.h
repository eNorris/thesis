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

    int xMesh;
    int yMesh;
    int zMesh;

    std::vector<float> xIndex;
    std::vector<float> yIndex;
    std::vector<float> zIndex;

    std::vector<float> dx;
    std::vector<float> dy;
    std::vector<float> dz;

    std::vector<float> DA;  // Area of the yz plane a ray sees
    std::vector<float> DB;  // Area of the xz plane a ray sees
    std::vector<float> DC;  // Area of the xy plane a ray sees

    std::vector<float> Axy;  // Replaces DA (Alreadys has the x2 factored in
    std::vector<float> Ayz;  // Replaces DB
    std::vector<float> Axz;  // Replaces DC

                                             // mu xi eta
    std::vector<unsigned int> orderOctant1;  // + + +
    std::vector<unsigned int> orderOctant2;  // - + +
    std::vector<unsigned int> orderOctant3;  // - - +
    std::vector<unsigned int> orderOctant4;  // + - +
    std::vector<unsigned int> orderOctant5;  // + + -
    std::vector<unsigned int> orderOctant6;  // - + -
    std::vector<unsigned int> orderOctant7;  // - - -
    std::vector<unsigned int> orderOctant8;  // + - -

    std::vector<float> vol;

    std::vector<unsigned short> zoneId;

    void load(const Config *config, const Quadrature *quad);

    int voxelCount() const;
    int xjmp() const;
    int yjmp() const;

    std::vector<unsigned int> &getOctantOrder(const float mu, const float xi, const float eta);

public slots:
    void remesh(int xelems, int yelems, int zelems, const Config *config, const Quadrature *quad);

private:
    bool insideBox(int x, int y, int z, int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);
    bool insideTightBox(int x, int y, int z, int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);
};

#endif // MESH_H
