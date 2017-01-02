#include "nuclidedata.h"

#include <QDebug>

NuclideData::NuclideData() :
    directory(),
    bondarenko1(),
    bondarenko2(),
    bondarenko3(),
    bondarenko4(),
    resonanceParams(),
    nAvgXs(),
    nScatter1(),
    nScatter2(),
    nScatter3(),
    gProduction1(),
    gProduction2(),
    gAvgXs(),
    gScatter1(),
    gScatter2()
{

}

NuclideData::~NuclideData()
{
    for(unsigned int i = 0; i  < nScatter2.size(); i++)
        if(nScatter2[i] != NULL)
            delete nScatter2[i];

    for(unsigned int i = 0; i  < nScatter3.size(); i++)
        if(nScatter3[i] != NULL)
            delete nScatter3[i];

    for(unsigned int i = 0; i  < gProduction2.size(); i++)
        if(gProduction2[i] != NULL)
            delete gProduction2[i];

    for(unsigned int i = 0; i  < gScatter2.size(); i++)
        if(gScatter2[i] != NULL)
            delete gScatter2[i];
}

bool NuclideData::parse(ifstream &binfile, int nGroups, int gGroups)
{
    directory.parse(binfile);

    int nsig0 = directory.getBondarenkoSig0Count();
    int nt = directory.getBondarenkoTempCount();
    int nbond = directory.getBondarenkoCount();

    int n2d = directory.getScatterNeutronProcCount();
    int g2d = directory.getScatterGammaProcCount();
    int n2g = directory.getNeutronToGammaProcCount();

    int nres = directory.getResolvedGroups();
    int nunr = directory.getUnresolvedGroups();

    int nAvgProc = directory.getAveragedNeutronProcCount();
    int gAvgProc = directory.getAveragedGammaProcCount();

    // Read the Bondarenko data
    if(nbond != 0)
    {
        bondarenko1.parse(binfile, nsig0, nt);
        bondarenko2.parse(binfile, nbond);

        size_t numTemp = bondarenko1.getTempCount();
        size_t numSig0 = bondarenko1.getSig0Count();
        const std::vector<int> &nf = bondarenko2.getNf();
        const std::vector<int> &nl = bondarenko2.getNl();

        bondarenko3.resize(nbond);
        bondarenko4.resize(nbond);

        for(int i = 0; i < nbond; i++)
            bondarenko3[i].parse(binfile, nf[i], nl[i]);

        for(int i = 0; i < nbond; i++)
            bondarenko4[i].parse(binfile, nf[i], nl[i], numTemp, numSig0);

    }

    // Read the resonance parameter data
    if(nres != 0 || nunr != 0)
        resonanceParams.parse(binfile, nres, nunr);

    // Read the neutron xs data
    nAvgXs.parse(binfile, nAvgProc, nGroups);

    // Read the neutron scatter data
    if(n2d > 0)
    {
        nScatter1.parse(binfile, n2d);

        for(int i = 0; i < n2d; i++)
        {
            const float procTempCount = nScatter1.getTemp()[i];
            const float procPlCount = nScatter1.getNlList()[i];
            const int procMaxLen = nScatter1.getLList()[i];

            if(procTempCount > 0)
            {
                AmpxRecordParserType11 *next11 = new AmpxRecordParserType11();
                next11->parse(binfile, procTempCount);
                nScatter2.push_back(next11);
            }

            for(int j = 0; j < fmax(procTempCount, 1)*(procPlCount+1); j++)
            {
                AmpxRecordParserType12 *next12 = new AmpxRecordParserType12();
                next12->parse(binfile, procMaxLen, false, nGroups);
                nScatter3.push_back(next12);
            }
        }
    }

    // Read the gamma production data
    if(n2g > 0)
    {
        gProduction1.parse(binfile, n2g);

        for(int i = 0; i < n2g; i++)
        {
            int procPlCount = gProduction1.getNlList()[i];

            for(int j = 0; j < procPlCount+1; j++)
            {
                AmpxRecordParserType12 *next12 = new AmpxRecordParserType12();
                next12->parse(binfile, -1, true, nGroups);
                gProduction2.push_back(next12);
            }
        }
    }

    // Read gamma xs data
    if(gAvgProc > 0)
        gAvgXs.parse(binfile, gAvgProc, gGroups);

    if(g2d > 0)
    {
        gScatter1.parse(binfile, g2d);

        for(int i = 0; i < g2d; i++)
        {
            int procPlCount = gScatter1.getNlList()[i];
            int procMaxLen = gScatter1.getLList()[i];

            for(int j = 0; j < procPlCount+1; j++)
            {
                AmpxRecordParserType12 *next12 = new AmpxRecordParserType12();
                next12->parse(binfile, procMaxLen, false, gGroups);
                gScatter2.push_back(next12);
            }
        }
    }

    return true;
}

AmpxRecordParserType12 *NuclideData::getGammaScatterMatrix(const int mt, const int nl_index) const
{
    const AmpxRecordParserType10 &scatdir = getGammaScatterDirectory();

    int mtindx = scatdir.getMtIndex(mt);
    if(mtindx < 0)
    {
        qDebug() << "Requested an invalid MT number!";
        return NULL;
    }

    int indx = 0;
    for(int i = 0; i < mtindx; i++)  // Iterate through all MT processes before the one of interest
    {
        indx += scatdir.getNlList()[i] + 1;  // Add the Legendre expansions in that process
    }

    int nlindx = nl_index;
    if(nl_index > (scatdir.getNlList()[mtindx]))
    {
        qDebug() << "Requested Legendre expansion is too large (" << nl_index << "/" << (scatdir.getNlList()[mtindx]-1) << ")";
        //nlindx = scatdir.getNlList()[mtindx]-1;
        return NULL;
    }

    indx += nlindx;  // Add the legendre expansions before the one of interest

    //qDebug() << "nuclidedata: 200: indx = " << indx;

    return getGammaScatterMatrices()[indx];
}
