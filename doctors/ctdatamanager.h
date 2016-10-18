#ifndef CTDATAREADER_H
#define CTDATAREADER_H

#include "mesh.h"
#include "quadrature.h"

class CtDataManager
{
protected:
    bool m_valid;
    //const static std::vector<int> hounsfieldRangePhantom19;
    //const static std::vector<int> hounsfieldRangePhantom19Elements;
    //const static std::vector<std::vector<float> > hounsfieldRangePhantom19Weights;

public:
    CtDataManager();
    ~CtDataManager();

    Mesh *parse16(int xbins, int ybins, int zbins, std::string filename);
    Mesh *ctNumberToHumanPhantom(Mesh *mesh);
    Mesh *ctNumberToQuickCheck(Mesh *mesh);
    //Mesh parse_256_256_64_16(std::string filename);

    //void setup();
};

#endif // CTDATAREADER_H
