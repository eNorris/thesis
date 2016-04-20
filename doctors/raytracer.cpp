
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

    qDebug() << "Running raytracer";

    float e1 = 1.0E-35;
    float e2 = 1.0E35;
    float e3 = 1.0E-8;
    float e4 = 2.5E-9;

    for(unsigned int iz = 0; iz < mesh->zElemCt; iz++)
        for(unsigned int iy = 0; iy < mesh->yElemCt; iy++)
            for(unsigned int ix = 0; ix < mesh->xElemCt; ix++)
                // TODO - for every source point as well
            {

            }

    return uflux;
}
