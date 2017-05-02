#ifndef MCNPWRITER_H
#define MCNPWRITER_H

#include <string>

class Mesh;
class SourceParams;

class McnpWriter
{
public:
    McnpWriter();
    ~McnpWriter();

    const static int MAX_BOUNDS = 100000;

    std::string generateSurfaceString(Mesh *m);
    std::string generateCellString(Mesh *m, bool fineDensity);
    std::string generateDataCards(SourceParams *p);
    std::string generatePhantom19MaterialString();
    std::string generateMeshTally(Mesh *m, SourceParams *xs);
    //std::string limit80Char(std::string s);

    void writeMcnp(std::string filename, Mesh *m, SourceParams *p, bool fineDensity);

protected:
    bool m_failFlag;

    std::string padFourDigitsZero(int v);
    std::string padFourDigitsSpace(int v);
    std::string xsurf(int xindx);
    std::string ysurf(int yindx);
    std::string zsurf(int zindx);
};

#endif // MCNPWRITER_H
