#ifndef CTDATAREADER_H
#define CTDATAREADER_H

#include "mesh.h"
#include "quadrature.h"

class CtDataManager
{
protected:
    bool m_valid;

public:
    CtDataManager();
    ~CtDataManager();

    Mesh *parse16(int xbins, int ybins, int zbins, std::string filename);
    Mesh *ctNumberToHumanPhantom(Mesh *mesh);
    Mesh *ctNumberToQuickCheck(Mesh *mesh);
};

#endif // CTDATAREADER_H
