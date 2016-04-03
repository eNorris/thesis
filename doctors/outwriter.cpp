#include "outwriter.h"

//#include <istream>
#include <fstream>
#include <vector>
#include <string>

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
    fout << mesh.xMesh << '\n';
    fout << mesh.yMesh << '\n';
    fout << mesh.zMesh << '\n';

    for(int i = 0; i < mesh.xMesh; i++)
        fout << mesh.xIndex[i] << '\n';
    for(int i = 0; i < mesh.yMesh; i++)
        fout << mesh.yIndex[i] << '\n';
    for(int i = 0; i < mesh.zMesh; i++)
        fout << mesh.zIndex[i] << '\n';

    if(mesh.xMesh * mesh.yMesh * mesh.zMesh != flux.size())
        qDebug() << "WARNING: OutWriter::writeScalarFlux: the mesh size did not match the data size";

    for(int i = 0; i < flux.size(); i++)
        fout << flux[i] << '\n';

    fout.flush();
    fout.close();
}

void OutWriter::writeScalarFluxMesh(std::string filename, const Mesh& mesh, const std::vector<float>& flux)
{
    std::ofstream fout;
    fout.open(filename.c_str());

    if(mesh.xMesh * mesh.yMesh * mesh.zMesh != flux.size())
        qDebug() << "WARNING: OutWriter::writeScalarFlux: the mesh size did not match the data size";

    for(int iz = 0; iz < mesh.zMesh; iz++)
    {
        fout << "\nz = " << iz << '\n';
        for(int iy = 0; iy < mesh.yMesh; iy++)
        {
            for(int ix = 0; ix < mesh.xMesh; ix++)
            {
                fout << flux[ix*mesh.yMesh*mesh.zMesh + iy*mesh.zMesh + iz] << '\t';
            }
            fout << '\n';
        }
    }

    fout.flush();
    fout.close();
}
