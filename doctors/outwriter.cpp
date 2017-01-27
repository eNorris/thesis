#include "outwriter.h"

#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

#include <QDebug>

#include "mesh.h"
#include "quadrature.h"
#include "xsection.h"

OutWriter::OutWriter()
{

}

void OutWriter::writeScalarFlux(std::string filename, const XSection& xs, const Mesh& mesh, const std::vector<float>& flux)
{
    std::ofstream fout;
    fout.open(filename.c_str());

    fout << 4 << '\n';
    fout << xs.groupCount() << '\n';
    fout << mesh.xElemCt << '\n';
    fout << mesh.yElemCt << '\n';
    fout << mesh.zElemCt << '\n';

    for(unsigned int i = 0; i < xs.groupCount(); i++)
        fout << i << '\n';
    for(unsigned int i = 0; i < mesh.xElemCt; i++)
        fout << mesh.xNodes[i] << '\n';
    for(unsigned int i = 0; i < mesh.yElemCt; i++)
        fout << mesh.yNodes[i] << '\n';
    for(unsigned int i = 0; i < mesh.zElemCt; i++)
        fout << mesh.zNodes[i] << '\n';

    if(mesh.voxelCount()*xs.groupCount() != flux.size())
        qDebug() << "WARNING: OutWriter::writeScalarFlux: the mesh size did not match the data size";

    for(unsigned int i = 0; i < flux.size(); i++)
        fout << flux[i] << '\n';

    fout.flush();
    fout.close();
}
/*
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
*/

void OutWriter::writeAngularFluxMesh(std::string filename, const XSection &xs, const Quadrature &quad, const Mesh &mesh, const std::vector<float> &flux)
{
    std::ofstream fout;
    fout.open(filename.c_str());

    if(xs.groupCount() * quad.angleCount() * mesh.xElemCt * mesh.yElemCt * mesh.zElemCt != flux.size())
        qCritical() << "WARNING: OutWriter::writeScalarFlux: the mesh size did not match the data size";

    fout << "5\n";
    fout << xs.groupCount() << '\n';
    fout << quad.angleCount() << '\n';
    fout << mesh.xElemCt << '\n';
    fout << mesh.yElemCt << '\n';
    fout << mesh.zElemCt << '\n';

    for(unsigned int ie = 0; ie < xs.groupCount(); ie++)
        fout << ie << '\n';

    for(unsigned int ia = 0; ia < quad.angleCount(); ia++)
        fout << ia << '\n';

    for(unsigned int ix = 0; ix < mesh.xElemCt; ix++)
        fout << mesh.xNodes[ix] << '\n';

    for(unsigned int iy = 0; iy < mesh.yElemCt; iy++)
        fout << mesh.yNodes[iy] << '\n';

    for(unsigned int iz = 0; iz < mesh.zElemCt; iz++)
        fout << mesh.zNodes[iz] << '\n';

    for(unsigned int i = 0; i < flux.size(); i++)
        fout << flux[i] << '\n';

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

void OutWriter::writeFloatArrays(std::string filename, const std::vector<std::vector<float> >& arry)
{
    std::ofstream fout;
    fout.open(filename.c_str());

    fout << std::fixed;
    fout << std::setprecision(6);

    for(unsigned int indx = 0; indx < arry[0].size(); indx++)
    {
        for(unsigned int ai = 0; ai < arry.size(); ai++)
        {
            fout << arry[ai][indx] << '\t';
        }
        fout << '\n';
    }

    fout.flush();
    fout.close();
}


