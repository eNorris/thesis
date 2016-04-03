#ifndef OUTWRITER_H
#define OUTWRITER_H

#include <vector>
#include <string>

class Mesh;

class OutWriter
{
public:
    OutWriter();

    static void writeScalarFlux(std::string filename, const Mesh& mesh, const std::vector<float>& flux);
    static void writeScalarFluxMesh(std::string filename, const Mesh& mesh, const std::vector<float>& flux);


};

#endif // OUTWRITER_H
