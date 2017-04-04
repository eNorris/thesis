#include "outwriter.h"

#include <fstream>
#include <vector>
#include <string>
//#include <iomanip>

//#include <QDebug>

#include "mesh.h"
#include "quadrature.h"
#include "xsection.h"

OutWriter::OutWriter()
{

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
        std::cout << "WARNING: OutWriter::writeZoneId: the mesh size did not match the data size" << std::endl;
        //qDebug() << "WARNING: OutWriter::writeZoneId: the mesh size did not match the data size";

    for(unsigned int i = 0; i < mesh.zoneId.size(); i++)
        fout << mesh.zoneId[i] << '\n';

    fout.flush();
    fout.close();
}

void OutWriter::writeFloatArrays(std::string filename, const std::vector<std::vector<float> >& arry)
{
    std::ofstream fout;
    fout.open(filename.c_str());

    //fout << std::fixed;
    //fout << std::setprecision(6);

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


