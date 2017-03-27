#ifndef OUTWRITER_H
#define OUTWRITER_H

#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <QDebug>

#include <iostream>
//#include <cstdarg>  // For variadic templates

#include "globals.h"

class Mesh;
class Quadrature;
class XSection;

class OutWriter
{
public:
    OutWriter();

    template<typename T>
    static void writeScalarFlux(std::string filename, const XSection &xs, const Mesh& mesh, const std::vector<T>& flux);

    template<typename T>
    static void writeAngularFlux(std::string filename, const XSection &xs, const Quadrature &quad, const Mesh &mesh, const std::vector<T>& flux);

    static void writeZoneId(std::string filename, const Mesh& mesh);

    template<typename T>
    static void writeArray(std::string filename, const std::vector<T>& arry);

    template<typename T1, typename T2>
    static void writeArray2(std::string filename, const std::vector<T1>& arry1, const std::vector<T2>& arry2);

    template<typename T1, typename T2, typename T3>
    static void writeArray3(std::string filename, const std::vector<T1>& arry1, const std::vector<T2>& arry2, const std::vector<T3>& arry3);

    static void writeFloatArrays(std::string filename, const std::vector<std::vector<float> >& arry);

};

#include "xsection.h"
#include "mesh.h"
#include "quadrature.h"

template<typename T>
void OutWriter::writeScalarFlux(std::string filename, const XSection& xs, const Mesh& mesh, const std::vector<T>& flux)
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

template<typename T>
void OutWriter::writeAngularFlux(std::string filename, const XSection &xs, const Quadrature &quad, const Mesh &mesh, const std::vector<T> &flux)
{
    std::ofstream fout;
    fout.open(filename.c_str());

    if(xs.groupCount() * quad.angleCount() * mesh.voxelCount() != flux.size())
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

template<typename T>
void OutWriter::writeArray(std::string filename, const std::vector<T>& arry)
{
    std::cout << "Writing 1D data to " << filename << std::endl;
    std::ofstream fout;
    fout.open(filename.c_str());

    //fout << std::fixed;
    //fout << std::setprecision(6);

    for(int i = 0; i < arry.size(); i++)
        fout << arry[i] << '\n';

    fout.flush();
    fout.close();
    std::cout << "Finished writing 1D data" << std::endl;
}

template<typename T1, typename T2>
void OutWriter::writeArray2(std::string filename, const std::vector<T1>& arry1, const std::vector<T2>& arry2)
{
    std::ofstream fout;
    fout.open(filename.c_str());

    fout << std::fixed;
    fout << std::setprecision(6);

    if(arry1.size() != arry2.size())
    {
        qDebug() << __FILE__ << ": " << __LINE__ << ": arrays sizes do not match";
    }

    int maxtimes = MIN(arry1.size(), arry2.size());

    for(int i = 0; i < maxtimes; i++)
        fout << arry1[i] << '\t' << arry2[i] << '\n';

    fout.flush();
    fout.close();
}

template<typename T1, typename T2, typename T3>
void OutWriter::writeArray3(std::string filename, const std::vector<T1>& arry1, const std::vector<T2>& arry2, const std::vector<T3>& arry3)
{
    std::ofstream fout;
    fout.open(filename.c_str());

    fout << std::fixed;
    fout << std::setprecision(6);

    if(arry1.size() != arry2.size() || arry1.size() != arry3.size())
    {
        qDebug() << __FILE__ << ": " << __LINE__ << ": arrays sizes do not match";
    }

    for(int i = 0; i < arry1.size(); i++)
        fout << arry1[i] << '\t' << arry2[i] << '\t' << arry3[i] << '\n';

    fout.flush();
    fout.close();
}



#endif // OUTWRITER_H
