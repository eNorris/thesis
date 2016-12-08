#ifndef OUTWRITER_H
#define OUTWRITER_H

#include <vector>
#include <string>
//#include <iostream>
#include <fstream>
#include <iomanip>
#include <QDebug>

#include <iostream>

#include <cstdarg>  // For variadic templates

class Mesh;

//template<typename T>
//static void writeArray(std::string filename, const std::vector<T>& arry);

//template<typename T1, typename T2>
//static void writeArray2(std::string filename, const std::vector<T1>& arry1, const std::vector<T2>& arry2);

class OutWriter
{
public:
    OutWriter();

    static void writeScalarFlux(std::string filename, const Mesh& mesh, const std::vector<float>& flux);
    static void writeScalarFluxMesh(std::string filename, const Mesh& mesh, const std::vector<float>& flux);

    static void writeZoneId(std::string filename, const Mesh& mesh);

    //static void writeArray(std::string filename, const std::vector<float>& arry);

    template<typename T>
    static void writeArray(std::string filename, const std::vector<T>& arry);

    template<typename T1, typename T2>
    static void writeArray2(std::string filename, const std::vector<T1>& arry1, const std::vector<T2>& arry2);

    template<typename T1, typename T2, typename T3>
    static void writeArray3(std::string filename, const std::vector<T1>& arry1, const std::vector<T2>& arry2, const std::vector<T3>& arry3);
    //static void writeArray(std::string filename, const std:::vector<float>& arry);

    static void writeFloatArrays(std::string filename, const std::vector<std::vector<float> >& arry);

    //template<typename... types>
    //void writeArrayT(std::string filename, const std::vector<types...>& args);

};

template<typename T>
void OutWriter::writeArray(std::string filename, const std::vector<T>& arry)
{
    std::cout << "Writing 1D data to " << filename << std::endl;
    std::ofstream fout;
    fout.open(filename.c_str());

    fout << std::fixed;
    fout << std::setprecision(6);

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

/*
template<typename... types>
void writeArrayT(std::string filename, const std::vector<types...>& vecs)
{
    std::ofstream fout;
    fout.open(filename.c_str());

    fout << std::fixed;
    fout << std::setprecision(6);

    va_list args;
    va_start(args, vecs);

    //if(arry1.size() != arry2.size() || arry1.size() != arry3.size())
    //{
    //    qDebug() << __FILE__ << ": " << __LINE__ << ": arrays sizes do not match";
    //}

    for(int i = 0; i < arry1.size(); i++)
        fout << arry1[i] << '\t' << arry2[i] << '\t' << arry3[i] << '\n';

    va_end(args);

    fout.flush();
    fout.close();
}
*/

#endif // OUTWRITER_H
