#include "outwriter.h"

//#include <istream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

#include <QDebug>

#include "mesh.h"

OutWriter::OutWriter()
{

}

void OutWriter::writeScalarFlux(std::string filename, const Mesh& mesh, const std::vector<float>& flux)
{
    std::ofstream fout;
    fout.open(filename.c_str());

    fout << 3 << '\n';
    fout << mesh.xElemCt << '\n';
    fout << mesh.yElemCt << '\n';
    fout << mesh.zElemCt << '\n';

    for(unsigned int i = 0; i < mesh.xElemCt; i++)
        fout << mesh.xNodes[i] << '\n';
    for(unsigned int i = 0; i < mesh.yElemCt; i++)
        fout << mesh.yNodes[i] << '\n';
    for(unsigned int i = 0; i < mesh.zElemCt; i++)
        fout << mesh.zNodes[i] << '\n';

    if(mesh.voxelCount() != flux.size())
        qDebug() << "WARNING: OutWriter::writeScalarFlux: the mesh size did not match the data size";

    for(unsigned int i = 0; i < flux.size(); i++)
        fout << flux[i] << '\n';

    fout.flush();
    fout.close();
}

void OutWriter::writeScalarFluxMesh(std::string filename, const Mesh& mesh, const std::vector<float>& flux)
{
    std::ofstream fout;
    fout.open(filename.c_str());

    if(mesh.xElemCt * mesh.yElemCt * mesh.zElemCt != flux.size())
        qDebug() << "WARNING: OutWriter::writeScalarFlux: the mesh size did not match the data size";

    for(unsigned int iz = 0; iz < mesh.zElemCt; iz++)
    {
        fout << "\nz = " << iz << '\n';
        for(unsigned int iy = 0; iy < mesh.yElemCt; iy++)
        {
            for(unsigned int ix = 0; ix < mesh.xElemCt; ix++)
            {
                fout << flux[ix*mesh.yElemCt*mesh.zElemCt + iy*mesh.zElemCt + iz] << '\t';
            }
            fout << '\n';
        }
    }

    fout.flush();
    fout.close();
}

void OutWriter::writeZoneId(std::string filename, const Mesh& mesh)
{
    std::ofstream fout;
    fout.open(filename.c_str());

    fout << 3 << '\n';
    fout << mesh.xElemCt << '\n';
    fout << mesh.yElemCt << '\n';
    fout << mesh.zElemCt << '\n';

    fout << std::fixed;
    fout << std::setprecision(6);

    for(unsigned int i = 0; i < mesh.xElemCt; i++)
        fout << mesh.xNodes[i] << '\n';
    for(unsigned int i = 0; i < mesh.yElemCt; i++)
        fout << mesh.yNodes[i] << '\n';
    for(unsigned int i = 0; i < mesh.zElemCt; i++)
        fout << mesh.zNodes[i] << '\n';

    if(mesh.voxelCount() != mesh.zoneId.size())
        qDebug() << "WARNING: OutWriter::writeZoneId: the mesh size did not match the data size";

    for(unsigned int i = 0; i < mesh.zoneId.size(); i++)
        fout << mesh.zoneId[i] << '\n';

    fout.flush();
    fout.close();
}
