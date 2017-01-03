#ifndef NUCLIDEDATA_H
#define NUCLIDEDATA_H

#include "ampxrecordparsers.h"
#include <vector>

class NuclideData
{
protected:
    AmpxRecordParserType3 directory;

    AmpxRecordParserType5 bondarenko1;
    AmpxRecordParserType6 bondarenko2;
    std::vector<AmpxRecordParserType7> bondarenko3;
    std::vector<AmpxRecordParserType8> bondarenko4;

    AmpxRecordParserType4 resonanceParams;

    AmpxRecordParserType9 nAvgXs;

    AmpxRecordParserType10 nScatter1;
    std::vector<AmpxRecordParserType11*> nScatter2;
    std::vector<AmpxRecordParserType12*> nScatter3;

    AmpxRecordParserType10 gProduction1;
    std::vector<AmpxRecordParserType12*> gProduction2;

    AmpxRecordParserType9 gAvgXs;

    AmpxRecordParserType10 gScatter1;
    std::vector<AmpxRecordParserType12*> gScatter2;

public:
    NuclideData();
    ~NuclideData();

    bool parse(ifstream &binfile, int nGroups, int gGroups);

    const AmpxRecordParserType3 &getDirectory() const { return directory; }
    const AmpxRecordParserType5 &getBondarenko1() const { return bondarenko1; }

    const AmpxRecordParserType9 &getNeutronXs() const { return nAvgXs; }

    const AmpxRecordParserType9 &getGammaXs() const { return gAvgXs; }

    const AmpxRecordParserType10 &getNeutronScatterDirectory() const { return nScatter1; }
    const std::vector<AmpxRecordParserType11*> &getNeutronScatterTemperatures() const { return nScatter2; }
    const std::vector<AmpxRecordParserType12*> &getNeutronScatterMatrices() const { return nScatter3; }

    const AmpxRecordParserType10 &getGammaScatterDirectory() const { return gScatter1; }
    const std::vector<AmpxRecordParserType12*> &getGammaScatterMatrices() const { return gScatter2; }
    AmpxRecordParserType12 *getGammaScatterMatrix(const int mt, const int nl) const;
};

#endif // NUCLIDEDATA_H
