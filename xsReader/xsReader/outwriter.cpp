#include "outwriter.h"

#include <QDebug>
#include <iostream>

#include "nuclidedata.h"

OutWriter::OutWriter(std::string filename) : m_filename(filename)
{
    m_fout.open(filename.c_str());
}

OutWriter::~OutWriter()
{
    m_fout.close();
}

void OutWriter::writeGammaScatterMatrix(const std::vector<float> &e, NuclideData *nuc, int mt)
{
    m_fout << "3\n";
    m_fout << (e.size()-1) << "\n" << (e.size()-1) << "\n";

    if(e.size()-1 != 19)
    {
        qDebug() << "Illegal energy size (" << e.size() << "/48)";
        return;
    }

    const AmpxRecordParserType10 &scatdir = nuc->getGammaScatterDirectory();

    int mtindx = scatdir.getMtIndex(mt);
    if(mtindx < 0)
    {
        qDebug() << "OutWriter: 33: MT value " << mt << " was illegal";
        return;
    }

    // The order is stored, add 1 to get number of elements
    int nlsize = scatdir.getNlList()[mtindx]+1;

    m_fout << nlsize << '\n';

    for(unsigned int i = 0; i < e.size()-1; i++)  // E
        m_fout << e[i] << '\n';

    for(unsigned int i = 0; i < e.size()-1; i++)  // E'
        m_fout << e[i] << '\n';

    for(int i = 0; i < nlsize; i++) // Nl
        m_fout << i << '\n';

    for(unsigned int i = 0; i < e.size()-1; i++)  // E index
        for(unsigned int j = 0; j < e.size()-1; j++)  // E' index
            for(int k = 0; k < nlsize; k++)  // Nl index
                m_fout << nuc->getGammaScatterMatrix(mt, k)->getXs(i, j) << '\n';

    for(unsigned int i = 0; i < e.size()-1; i++)  // E index
    {
        for(unsigned int j = 0; j < e.size()-1; j++)  // E' index
            std::cout << nuc->getGammaScatterMatrix(mt, 0)->getXs(i, j) << "\t";
        std::cout << std::endl;
    }

    //const std::vector<int> &mts = scatdir.getMtList();

    //for(unsigned int i = 0; i < mts.size(); i++)
    //    qDebug() << mts[i] << "  " << (be9scatdir.getNlList()[i] + 1);



    //const std::vector<AmpxRecordParserType12*> &be9scats = be9->getGammaScatterMatrices();

    //qDebug() << "Total scatter matrices in Be9: " << be9scats.size();

    //AmpxRecordParserType12 *be9scat = be9->getGammaScatterMatrix(504, 6);
}
