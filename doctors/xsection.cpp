#include "xsection.h"

//#define _USE_MATH_DEFINES
#include <cmath>

#include <QDebug>
#include <QMessageBox>
#include <fstream>

#include "xs_reader/ampxparser.h"
#include "materialutils.h"
#include "quadrature.h"

XSection::XSection() : m_elements(NULL), m_weights(NULL), m_groups(0), m_matsLoaded(0)
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
    //qDebug() << "This function is now depricated until flux moments are implemented";

    if(matid > m_matsLoaded || gSrcIndx >= m_groups || gSinkIndx >= m_groups || n > m_pn)
    {
        qDebug() << "Illegal index value";
        qDebug() << "Violated matid > m_matsLoaded || gSrcIndx >= m_groups || gSinkIndx >= m_groups || n > m_pn";
        qDebug() <<  matid << "> " << m_matsLoaded << " || " << gSrcIndx << " >= " << m_groups << " || " << gSinkIndx << " >= " << m_groups << " || " << n << " > " << m_pn;
        return -1.0;
    }
    const int pnCount = m_pn+1;
    return m_scat2d[matid*m_groups*m_groups*pnCount + gSrcIndx*m_groups*pnCount + gSinkIndx*pnCount + n];
}

float XSection::scatxsKlein(const int matid, const int gSrcIndx, const int gSinkIndx, const int aSrcIndx, const int aSinkIndx) const
{
    if(matid > m_matsLoaded || gSrcIndx >= m_groups || gSinkIndx >= m_groups || aSrcIndx >= m_angles || aSinkIndx >= m_angles)
    {
        QString errmsg = QString("Indexing subscript went out of bounds in scatxsKlein()");
        //errmsg += "The table has room for " + QString::number(m_mats) + " materials.";
        QMessageBox::warning(NULL, "Indexing Error", errmsg, QMessageBox::Close);
        //qDebug() <<  matid << "> " << m_matsLoaded << " || " << gSrcIndx << " >= " << m_groups << " || " << gSinkIndx << " >= " << m_groups << " || " << n << " > " << m_pn;
        return -1.0;
    }
    //const int pnCount = m_pn+1;
    return m_scatKlein[matid*m_groups*m_groups*m_angles*m_angles + gSrcIndx*m_groups*m_angles*m_angles + gSinkIndx*m_angles*m_angles + aSrcIndx*m_angles + aSinkIndx];
}

bool XSection::allocateMemory(const unsigned int materialCount, const unsigned int groupCount, const unsigned int pn)
{
    m_tot1d.clear();
    m_scat1d.clear();
    m_scat2d.clear();

    m_mats = materialCount;
    m_groups = groupCount;
    m_pn = pn;
    size_t floats1d = materialCount * groupCount;
    size_t floats2d = materialCount * groupCount * groupCount * (m_pn+1);

    try{
        m_tot1d.resize(floats1d);
    }
    catch(std::bad_alloc &bad){
        QString errmsg = QString("bad_alloc caught during XS initialization of the 1d total xs data, requested ");
        errmsg += QString::number(floats1d * sizeof(float));
        errmsg += " bytes. Reported error: ";
        errmsg += bad.what();
        QMessageBox::warning(NULL, "Out of Memory", errmsg, QMessageBox::Close);

        return false;
    }

    try{
        m_scat1d.resize(floats1d);
    }
    catch(std::bad_alloc &bad){
        QString errmsg = QString("bad_alloc caught during XS initialization of the 1d scatter xs data, requested ");
        errmsg += QString::number(floats1d * sizeof(float));
        errmsg += " bytes. Reported error: ";
        errmsg += bad.what();
        QMessageBox::warning(NULL, "Out of Memory", errmsg, QMessageBox::Close);

        return false;
    }

    try{
        m_scat2d.resize(floats2d);
    }
    catch(std::bad_alloc &bad){
        QString errmsg = QString("bad_alloc caught during XS initialization of the 2d scatter xs data, requested ");
        errmsg += QString::number(floats2d * sizeof(float));
        errmsg += " bytes. Reported error: ";
        errmsg += bad.what();
        QMessageBox::warning(NULL, "Out of Memory", errmsg, QMessageBox::Close);
        return false;
    }

    return true;
}

bool XSection::allocateMemory(const unsigned int groupCount, const unsigned int pn)
{
    if(m_elements == NULL || m_weights == NULL)
        return false;
    return allocateMemory(m_weights->size()+1, groupCount, pn);
}

bool XSection::addMaterial(const std::vector<int> &z, const std::vector<float> &w, const AmpxParser *p)
{
    if(m_matsLoaded >= m_mats)
    {
        QString errmsg = QString("There was an internal error, the xs table is already full but another material was added to it.");
        errmsg += "The table has room for " + QString::number(m_mats) + " materials.";
        QMessageBox::warning(NULL, "Internal Error", errmsg, QMessageBox::Close);
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
            const int pnCount = m_pn+1;
            for(int n = 0; n <= m_pn; n++)
            {
                AmpxRecordParserType12 *gxsc = d->getGammaScatterMatrix(MT_GAMMA_COHERENT_SCATTER, n);
                if(gxsc != NULL)
                    for(int iesrc = 0; iesrc < m_groups; iesrc++)
                    {
                        for(int iesnk = 0; iesnk < m_groups; iesnk++)
                        {
                            m_scat2d[m_matsLoaded*m_groups*m_groups*pnCount + iesrc*m_groups*pnCount + iesnk*pnCount + n] += gxsc->getXs(iesrc, iesnk)*afrac;
                        }
                    }

                gxsc = d->getGammaScatterMatrix(MT_GAMMA_INCOHERENT_SCATTER, n);
                if(gxsc != NULL)
                    for(int iesrc = 0; iesrc < m_groups; iesrc++)
                    {
                        for(int iesnk = 0; iesnk < m_groups; iesnk++)
                        {
                            m_scat2d[m_matsLoaded*m_groups*m_groups*pnCount + iesrc*m_groups*pnCount + iesnk*pnCount + n] += gxsc->getXs(iesrc, iesnk)*afrac;
                        }
                    }
            }
        }
        else  // Look at all isotopes of z[i]
        {
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
                    const int pnCount = m_pn + 1;
                    for(int n = 0; n < pnCount; n++)
                    {
                        AmpxRecordParserType12 *gxsc = d->getGammaScatterMatrix(MT_GAMMA_COHERENT_SCATTER, n);
                        if(gxsc != NULL)
                            for(int iesrc = 0; iesrc < m_groups; iesrc++)
                            {
                                for(int iesnk = 0; iesnk < m_groups; iesnk++)
                                {
                                    m_scat2d[m_matsLoaded*m_groups*m_groups*pnCount + iesrc*m_groups*pnCount + iesnk*pnCount + n] += gxsc->getXs(iesrc, iesnk)*afrac*MaterialUtils::naturalAbundances[elemIndx][j];
                                }
                            }

                        gxsc = d->getGammaScatterMatrix(MT_GAMMA_INCOHERENT_SCATTER, n);
                        if(gxsc != NULL)
                            for(int iesrc = 0; iesrc < m_groups; iesrc++)
                            {
                                for(int iesnk = 0; iesnk < m_groups; iesnk++)
                                {
                                    m_scat2d[m_matsLoaded*m_groups*m_groups*pnCount + iesrc*m_groups*pnCount + iesnk*pnCount + n] += gxsc->getXs(iesrc, iesnk)*afrac*MaterialUtils::naturalAbundances[elemIndx][j];
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

                for(int n = 0; n < m_pn; n++)
                {
                    for(int iesrc = 0; iesrc < m_groups; iesrc++)
                    {
                        for(int iesnk = 0; iesnk < m_groups; iesnk++)
                        {
                            m_scat2d[m_matsLoaded*m_groups*m_groups*m_pn + iesrc*m_groups*m_pn + iesnk*m_pn + n] /= weightCovered;
                        }
                    }
                }
            }
            else
            {
                // No data was found for element
                if(z[i] > MaterialUtils::elementNames.size())
                {
                    QString errmsg = QString("Requested element Z=") + QString::number(z[i]) + ". This element does not exist.";
                    QMessageBox::warning(NULL, "No Such Element", errmsg, QMessageBox::Close);
                }
                else
                {
                    QString element = QString::fromStdString(MaterialUtils::elementNames[z[i]]);
                    QString errmsg = QString("No data was found for ") + element + "(Z=" + QString::number(z[i]) + "). ";
                    errmsg += "Neither natural nor isotopic information was found in ";
                    errmsg += p->getFilename() + ".";
                    QMessageBox::warning(NULL, "No Data Found", errmsg, QMessageBox::Close);
                }
            }
        }  // if natural not in library
    } // for each z[i]

    // This material is now loaded
    m_matsLoaded++;

    return true;
}

bool XSection::addAll(AmpxParser *parser){

    if(m_elements == NULL || m_weights == NULL)
        return false;

    bool allPassed = true;

    gbounds = parser->getGammaEnergy();

    // Add the materials to the xs library
    for(unsigned int i = 0; i < m_weights->size(); i++)
        if(allPassed)
            allPassed &= addMaterial(*m_elements, (*m_weights)[i], parser);

    // The last material is empty and should never be used
    if(allPassed)
        allPassed &= addMaterial(std::vector<int>{}, std::vector<float>{}, parser);

    return allPassed;
}

bool XSection::buildKleinTable(Quadrature *quad)
{
    m_angles = quad->angleCount();

    m_scatKlein.clear();

    // Allocate memory
    try{
        m_scatKlein.resize(m_matsLoaded * m_groups * m_groups * m_angles * m_angles);
    }
    catch(std::bad_alloc &bad){
        QString errmsg = QString("bad_alloc caught during XS initialization of the Klein-Nishina scatter xs data, requested ");
        errmsg += QString::number(m_matsLoaded * m_groups * m_groups * m_angles * m_angles * sizeof(float));
        errmsg += " bytes. Reported error: ";
        errmsg += bad.what();
        QMessageBox::warning(NULL, "Out of Memory", errmsg, QMessageBox::Close);
        return false;
    }

    std::vector<float> tmp_mu;
    tmp_mu.resize(m_scatKlein.size());

    // Once the data is loaded, the Klein-Nishina data table is built as well.
    unsigned int zjmp = m_groups * m_groups * m_angles * m_angles;
    unsigned int ejmp = m_groups * m_angles * m_angles;
    unsigned int epjmp = m_angles * m_angles;
    unsigned int ajmp = m_angles;

    const float mec2 = 0.511E6;  // Rest mass energy of electron in eV

    for(int zid = 0; zid < m_matsLoaded; zid++)
        for(int ie = 0; ie < m_groups; ie++)
            for(int iep = 0; iep < m_groups; iep++)
            {
                // Skip upscatter
                if(iep < ie)
                    continue;

                float sigE = scatxs2d(zid, ie, iep, 0); // sigma(E -> E')

                // Every ie -> iep follows the Klein Nishina formula:
                // sigma(iep/ie, mu) = K*q*(1 + q^2 - (1-mu^2)), q = iep/ie
                float K = 0;

                for(int ia = 0; ia < m_angles; ia++)
                    for(int iap = 0; iap < m_angles; iap++)
                    {
                        float mu = quad->mu[ia]*quad->mu[iap] + quad->eta[ia]*quad->eta[iap] + quad->zi[ia]*quad->zi[iap];
                        float E = (gbounds[ie] + gbounds[ie+1])/2;
                        //float Ep = (gbounds[iep] + gbounds[iep+1])/2;
                        //float q = Ep/E;
                        float q = 1/(1 + (E/mec2)*(1-mu));
                        m_scatKlein[zid*zjmp + ie*ejmp + iep*epjmp + ia*ajmp + iap] = q*(q*q + mu*mu);  //q*q*q + q*mu*mu;
                        K += m_scatKlein[zid*zjmp + ie*ejmp + iep*epjmp + ia*ajmp + iap] * quad->wt[ia] * quad->wt[iap];
                        tmp_mu[zid*zjmp + ie*ejmp + iep*epjmp + ia*ajmp + iap] = mu;
                    }

                // The integral over all Omega and Omega' must equal sigE

                for(int ia = 0; ia < m_angles; ia++)
                    for(int iap = 0; iap < m_angles; iap++)
                    {

                        m_scatKlein[zid*zjmp + ie*ejmp + iep*epjmp + ia*ajmp + iap] *= (sigE/K);
                    }
            }


    std::ofstream fout;
    fout.open("kleinlog.dat");
    for(unsigned int i = 0; i < m_scatKlein.size(); i++)
        fout << tmp_mu[i] << "\t" << m_scatKlein[i] << '\n';
    fout.close();
    /*
    std::vector<float> tst;
    tst.resize(m_groups);
    for(int ie = 0; ie < m_groups; ie++)
        for(int iep = 0; iep < m_groups; iep++)
            for(int ia = 0; ia < m_angles; ia++)
                for(int iap = 0; iap < m_angles; iap++)
                {
                    float q = m_scatKlein[2*zjmp + ie*ejmp + iep*epjmp + ia*ajmp + iap];
                    tst[ie] += m_scatKlein[2*zjmp + ie*ejmp + iep*epjmp + ia*ajmp + iap] * quad->wt[ia] * quad->wt[iap];
                }

    std::vector<float> tst2;
    std::vector<float> mu2;
    tst2.resize(m_angles);
    mu2.resize(m_angles);
    for(iep = 0; iep < m_groups; iep++)
        for(int ia = 0; ia < m_angles; ia++)
            for(int iap = 0; iap < m_angles; iap++)
            {
                //mu2[] = 5;
                //tst2[] = 5;
            }
            */

    // tst should be identical to scat1d
    qDebug() << "Done";

    return true;
}

bool XSection::setElements(const std::vector<int> &elem, const std::vector<std::vector<float> > &wt)
{
    for(unsigned int i = 0; i < elem.size(); i++)
    {
        if(elem.size() != wt[i].size())
        {
            QString errmsg = QString("setElements faild due to a size mismatch");
            QMessageBox::warning(NULL, "Internal Error", errmsg, QMessageBox::Close);
            return false;
        }
    }

    m_elements = &elem;
    m_weights = &wt;

    return true;
}
