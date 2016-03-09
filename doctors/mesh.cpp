#include "mesh.h"

#include <cmath>

Mesh::Mesh()
{

}

void Mesh::load(const Config &config)
{
    xmesh = 89;
    ymesh = 99;
    zmesh = ceil(5.0 / (2.0 * config.sourceFrontGap));


}
