#ifndef OUTWRITER_H
#define OUTWRITER_H

#include <string>
#include <fstream>
#include <vector>

class NuclideData;

class OutWriter
{
public:
    OutWriter(std::string filename);
    ~OutWriter();

    void writeGammaScatterMatrix(const std::vector<float> & e, NuclideData *nuc, int mt);

protected:
    std::string m_filename;
    std::ofstream m_fout;
};

#endif // OUTWRITER_H
