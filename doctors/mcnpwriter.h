#ifndef MCNPWRITER_H
#define MCNPWRITER_H

#include <string>

class Mesh;

class McnpWriter
{
public:
    McnpWriter();
    ~McnpWriter();

    const static int MAX_BOUNDS = 100000;

    std::string generateSurfaceString(Mesh *m);
    std::string generateCellString(Mesh *m);
    std::string generateDataCards();
    std::string generatePhantom19MaterialString();

    void writeMcnp(std::string filename, Mesh *m);

protected:
    bool m_failFlag;

    std::string padFourDigitsZero(int v);
    std::string padFourDigitsSpace(int v);
    std::string xsurf(int xindx);
    std::string ysurf(int yindx);
    std::string zsurf(int zindx);
};

#endif // MCNPWRITER_H
