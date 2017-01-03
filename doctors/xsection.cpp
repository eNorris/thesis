#include "xsection.h"

#include <cmath>

#include <QDebug>

#include "xs_reader/ampxparser.h"
#include "materialutils.h"

XSection::XSection() : m_groups(0), m_matsLoaded(0)
{

}

XSection::~XSection()
{

}

unsigned int XSection::groupCount() const
{
    return m_groups;
}

float XSection::scatXs1d(const int matid, const int g) const
{
    return m_scat1d[matid*m_groups + g];
}

float XSection::totXs1d(const int matid, const int g) const
{
    return m_tot1d[matid*m_groups + g];
}

float XSection::scatxs2d(const int matid, const int gSrcIndx, const int gSinkIndx, const int n) const
{
    if(matid > m_matsLoaded || gSrcIndx > (m_groups-1) || gSinkIndx > (m_groups-1) || n >= m_pns)
    {
        qDebug() << "XSection::scatxs2d(): 39: Illegal index value";
        return -1.0;
    }
    return m_scat2d[matid*m_groups*m_groups*m_pns + gSrcIndx*m_groups*m_pns + gSinkIndx*m_pns + n];
}

bool XSection::allocateMemory(const unsigned int materialCount, const unsigned int groupCount, const unsigned int pnCount)
{
    m_mats = materialCount;
    m_groups = groupCount;
    m_pns = pnCount;
    size_t bytes1d = materialCount * groupCount * sizeof(float);
    size_t bytes2d = materialCount * groupCount * groupCount * pnCount * sizeof(float);

    qDebug() << "Total 1d bytes: " << (2*bytes1d);
    qDebug() << "Total 2d bytes: " << bytes2d;

    try{
        m_tot1d.resize(bytes1d);
    }
    catch(std::bad_alloc &bad)
    {
        qDebug() << "bad_alloc caught during XS initialization of the 1d total xs data, requested " << bytes1d << " bytes: ";
        qDebug() << "Reported error: " << bad.what(); ;
        return false;
    }

    try{m_scat1d.resize(bytes1d);m_tot1d.resize(bytes1d);
    }
    catch(std::bad_alloc &bad)
    {
        qDebug() << "bad_alloc caught during XS initialization of the 1d scatter xs data, requested " << bytes1d << " bytes: ";
        qDebug() << "Reported error: " << bad.what(); ;
        return false;
    }

    try{
        m_scat2d.resize(bytes2d);
    }
    catch(std::bad_alloc &bad)
    {
        qDebug() << "bad_alloc caught during XS initialization of the 2d scatter xs data, requested " << bytes2d << " bytes: ";
        qDebug() << "Reported error: " << bad.what(); ;
        return false;
    }

    return true;
}

bool XSection::addMaterial(const std::vector<int> &z, const std::vector<float> &w, const AmpxParser *p)
{

    if(m_matsLoaded >= m_mats)
    {
        qDebug() << "XSection::addMaterial(): 93: Tried to add another cross section to a full table";
        qDebug() << "XSection::addMaterial(): 94: Max size = " << m_mats;
        return false;
    }

    std::vector<float> atom_frac = MaterialUtils::weightFracToAtomFrac(z, w);

    // For each z
    for(unsigned int i = 0; i < z.size(); i++)
    {

        float afrac = atom_frac[i];
        unsigned int elemIndx = z[i] - 1;

        // If there is a natural isotope in the library, use it
        int naturalZAID = z[i]*1000;

        int naturalIndx = p->getIndexByZaid(naturalZAID);

        if(naturalIndx >= 0)
        {
            // Handle the scatter 1d xs
            NuclideData *d = p->getData(naturalIndx);
            const AmpxRecordParserType9 &gxs = d->getGammaXs();
            int scat1dIndexInc = gxs.getMtIndex(MT_GAMMA_INCOHERENT_SCATTER);
            int scat1dIndexCoh = gxs.getMtIndex(MT_GAMMA_COHERENT_SCATTER);

            // Scatter is the sum of coherent and incoherent
            if(scat1dIndexInc >= 0)
            {
                std::vector<float> scat1dArrayInc = gxs.getSigmaMt(scat1dIndexInc);
                for(int ei = 0; ei < m_groups; ei++)
                    m_scat1d[m_matsLoaded*m_groups + ei] += scat1dArrayInc[ei]*afrac;
            }

            if(scat1dIndexCoh >= 0)
            {
                std::vector<float> scat1dArrayCoh = gxs.getSigmaMt(scat1dIndexCoh);
                for(int ei = 0; ei < m_groups; ei++)
                    m_scat1d[m_matsLoaded*m_groups + ei] += scat1dArrayCoh[ei]*afrac;
            }

            // Handle the total 1d xs
            int tot1dIndex = gxs.getMtIndex(MT_GAMMA_TOTAL_INTERACTION);
            if(tot1dIndex >= 0)
            {
                std::vector<float> tot1dArray = gxs.getSigmaMt(tot1dIndex);
                for(int ei = 0; ei < m_groups; ei++)
                    m_tot1d[m_matsLoaded*m_groups + ei] += tot1dArray[ei]*afrac;
            }

            // Handle the scatter 2d xs
            for(int n = 0; n < m_pns; n++)
            {
                AmpxRecordParserType12 *gxsc = d->getGammaScatterMatrix(MT_GAMMA_COHERENT_SCATTER, n);
                if(gxsc != NULL)
                    for(int iesrc = 0; iesrc < m_groups; iesrc++)
                    {
                        for(int iesnk = 0; iesnk < m_groups; iesnk++)
                        {
                            m_scat2d[m_matsLoaded*m_groups*m_groups*m_pns + iesrc*m_groups*m_pns + iesnk*m_pns + n] += gxsc->getXs(iesrc, iesnk)*afrac;
                        }
                    }

                gxsc = d->getGammaScatterMatrix(MT_GAMMA_INCOHERENT_SCATTER, n);
                if(gxsc != NULL)
                    for(int iesrc = 0; iesrc < m_groups; iesrc++)
                    {
                        for(int iesnk = 0; iesnk < m_groups; iesnk++)
                        {
                            m_scat2d[m_matsLoaded*m_groups*m_groups*m_pns + iesrc*m_groups*m_pns + iesnk*m_pns + n] += gxsc->getXs(iesrc, iesnk)*afrac;
                        }
                    }
            }
        }
        else  // Look at all isitopes of z[i]
        {
            //qDebug() << "No natural isotope for z = " << z[i];
            float weightCovered = 0.0f;

            // Iterate through all known isotopes
            for(unsigned int j = 0; j < MaterialUtils::naturalIsotopes[elemIndx].size(); j++)
            {
                int isotopeZaid = MaterialUtils::naturalIsotopes[elemIndx][j] + z[i]*1000;
                int isotopeIndex = p->getIndexByZaid(isotopeZaid);
                if(isotopeIndex >= 0)
                {
                    weightCovered += MaterialUtils::naturalAbundances[elemIndx][j];

                    // Handle the scatter 1d xs
                    NuclideData *d = p->getData(isotopeIndex);
                    const AmpxRecordParserType9 &gxs = d->getGammaXs();
                    int scat1dIndexInc = gxs.getMtIndex(MT_GAMMA_INCOHERENT_SCATTER);
                    int scat1dIndexCoh = gxs.getMtIndex(MT_GAMMA_COHERENT_SCATTER);

                    // Scatter is the sum of coherent and incoherent
                    if(scat1dIndexInc >= 0)
                    {
                        const std::vector<float> &scat1dArrayInc = gxs.getSigmaMt(scat1dIndexInc);
                        for(int ei = 0; ei < m_groups; ei++)
                        {
                            //int indx = m_matsLoaded*m_groups + ei;
                            //int oindx = z[i];
                            m_scat1d[m_matsLoaded*m_groups + ei] += scat1dArrayInc[ei]*afrac*MaterialUtils::naturalAbundances[elemIndx][j];
                        }
                    }

                    if(scat1dIndexCoh >= 0)
                    {
                        const std::vector<float> &scat1dArrayCoh = gxs.getSigmaMt(scat1dIndexCoh);
                        for(int ei = 0; ei < m_groups; ei++)
                            m_scat1d[m_matsLoaded*m_groups + ei] += scat1dArrayCoh[ei]*afrac*MaterialUtils::naturalAbundances[elemIndx][j];
                    }

                    // Handle the total 1d xs
                    int tot1dIndex = gxs.getMtIndex(MT_GAMMA_TOTAL_INTERACTION);
                    if(tot1dIndex >= 0)
                    {
                        const std::vector<float> &tot1dArray = gxs.getSigmaMt(tot1dIndex);
                        for(int ei = 0; ei < m_groups; ei++)
                            m_tot1d[m_matsLoaded*m_groups + ei] += tot1dArray[ei]*afrac*MaterialUtils::naturalAbundances[elemIndx][j];
                    }

                    // Handle the scatter 2d xs
                    for(int n = 0; n < m_pns; n++)
                    {
                        AmpxRecordParserType12 *gxsc = d->getGammaScatterMatrix(MT_GAMMA_COHERENT_SCATTER, n);
                        if(gxsc != NULL)
                            for(int iesrc = 0; iesrc < m_groups; iesrc++)
                            {
                                for(int iesnk = 0; iesnk < m_groups; iesnk++)
                                {
                                    m_scat2d[m_matsLoaded*m_groups*m_groups*m_pns + iesrc*m_groups*m_pns + iesnk*m_pns + n] += gxsc->getXs(iesrc, iesnk)*afrac*MaterialUtils::naturalAbundances[elemIndx][j];
                                }
                            }

                        gxsc = d->getGammaScatterMatrix(MT_GAMMA_INCOHERENT_SCATTER, n);
                        if(gxsc != NULL)
                            for(int iesrc = 0; iesrc < m_groups; iesrc++)
                            {
                                for(int iesnk = 0; iesnk < m_groups; iesnk++)
                                {
                                    m_scat2d[m_matsLoaded*m_groups*m_groups*m_pns + iesrc*m_groups*m_pns + iesnk*m_pns + n] += gxsc->getXs(iesrc, iesnk)*afrac*MaterialUtils::naturalAbundances[elemIndx][j];
                                }
                            }
                    }
                } // if the isotope was found
            } // for every isotope

            if(weightCovered > 1E-6)
            {
                // Divide by the total weight covered by the library
                for(int ei = 0; ei < m_groups; ei++)
                {
                    m_scat1d[m_matsLoaded*m_groups + ei] /= weightCovered;
                    m_tot1d[m_matsLoaded*m_groups + ei] /= weightCovered;
                }

                for(int n = 0; n < m_pns; n++)
                {
                    for(int iesrc = 0; iesrc < m_groups; iesrc++)
                    {
                        for(int iesnk = 0; iesnk < m_groups; iesnk++)
                        {
                            m_scat2d[m_matsLoaded*m_groups*m_groups*m_pns + iesrc*m_groups*m_pns + iesnk*m_pns + n] /= weightCovered;
                        }
                    }
                }
            }
            else
            {
                qDebug() << "No isotopes of element " << z[i] << " found in data library";
                //qDebug() << "No isotopes of element " << z[i] << "(" << qPrintable(MaterialUtils::elementNames[z[i]]) << ")" << " found in data library";
            }
        }  // if natural not in library
    } // for each z[i]

    // This material is now loaded
    m_matsLoaded++;

    return true;
}

