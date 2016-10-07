#include "xsection.h"

#include <cmath>

#include <QDebug>

#include "xs_reader/ampxparser.h"
#include "materialutils.h"

XSection::XSection() : m_groups(0), m_matsLoaded(0)
{

}

//XSection::XSection(const Config *config) : m_groups(0)
//{
//    //load(config);
//}

XSection::~XSection()
{

}
/*
void XSection::load(const Config *config)
{
    if(config->njoy)
    {
        qDebug() << "Sorry, still under development";
    }
    else
    {
        qDebug() << "Reading XS locally";

        m_groups = config->igm;
        m_zids = 3;

        // TODO - What is this?
        //int t1 = round(config->mtm / (1 + config->isct));
        //int t2 = 1 + config->isct;
        //int t3 = 2 + config->ihm - config->ihs;

        int dim1 = round(config->mtm / (1 + config->isct));
        int dim2 = 1 + config->isct;
        int dim3 = 2 + config->ihm - config->ihs;

        m_dim1 = dim1*dim2*dim3;
        m_dim2 = dim2*dim3;
        m_dim3 = dim3;

        //msig.resize(m_groups * t1 * t2 * t3);
        m_scat2d.resize(m_groups * m_dim1 * m_dim2 * m_dim3);
        m_tot1d.resize(m_groups*m_zids);
        m_scat1d.resize(m_groups*m_groups*m_zids);

        //for(int i = 0; i < m_groups; i++)
        //    for(int j = 0; j < t1; j++)
        //        for(int k = 0; k < t2; k++)
        //            for(int m = 0; m < t3; m++)
        //                msig[i*t1*t2*t3 + j*t2*t3 + k*t3 + m] = config->xsection[((j-1) * (1+config->isct) + k)*4 + config->iht + m - 1];

        for(int i = 0; i < m_groups; i++)
            for(int j = 0; j < m_dim1; j++)
                for(int k = 0; k < m_dim2; k++)
                    for(int m = 0; m < m_dim3; m++)
                        m_scat2d[i*m_dim1 + j*m_dim2 + k*m_dim3 + m] = config->xsection[((j-1) * (1+config->isct) + k)*4 + config->iht + m - 1];

        for(int zid = 0; zid < m_zids; zid++)
            for(int Esrc = 0; Esrc < m_groups; Esrc++)
                for(int Etar = 0; Etar < m_groups; Etar++)
                    m_scat1d[zid*m_groups*m_groups + Esrc*m_groups + Etar] = config->xsScat[zid*m_groups*m_groups + Esrc*m_groups + Etar];

        for(int zid = 0; zid < m_zids; zid++)
            for(int E = 0; E < m_groups; E++)
                m_tot1d[zid*m_groups + E] = config->xsTot[zid*m_groups + E];
    }
}
*/

/*
int XSection::operator()(int grp, int d2, int d3, int d4) const
{
    unsigned int indx = grp*m_dim1 + d2*m_dim2 + d3*m_dim3 + d4;
    if(indx >= m_scat2d.size())
        qDebug() << "XSection indexing error: Accessed " << (indx+1) << "/" << m_scat2d.size();
    return m_scat2d[grp*m_dim1 + d2*m_dim2 + d3*m_dim3 + d4];
}
*/

unsigned int XSection::groupCount() const
{
    return m_groups;
}

float XSection::scatXs1d(const int zid, const int Esrc, const int Etar) const
{
    return m_scat1d[zid*m_groups*m_groups + Esrc*m_groups + Etar];
}

float XSection::totXs1d(const int zid, const int E) const
{
    return m_tot1d[zid*m_groups + E];
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
        qDebug() << "XSection::addMaterial(): 156: Tried to add another cross section to a full table";
        qDebug() << "Max size = " << m_mats;
        return false;
    }
    m_matsLoaded++;

    // First convert atom weight to atom fraction
    std::vector<float> atom_frac;

    float totalWeight = 0.0f;
    for(unsigned int i = 0; i < z.size(); i++)
    {
        atom_frac.push_back(w[i]/MaterialUtils::atomicMass[z[i]]);
        totalWeight += atom_frac[i];
    }

    for(unsigned int i = 0; i < atom_frac.size(); i++)
        atom_frac[i] /= totalWeight;

    // For each z
    for(unsigned int i = 0; i < z.size(); i++)
    {
        // If there is a natural isotope in the library, use it
        int naturalZAID = z[i]*1000;

        int naturalIndx = p->getIndexByZaid(naturalZAID);

        if(naturalIndx >= 0)
        {
            // Add the natrual composition to the material
            m_tot1d;
            m_scat1d;
            m_scat2d;
        }
        else
        {
            float weightCovered = 0.0f;
            // Iterate through all known isotopes
            for(unsigned int j = 0; j < MaterialUtils::naturalIsotopes[z[i]].size(); j++)
            {
                int isotopeZaid = MaterialUtils::naturalIsotopes[z[i]][j] + z[i]*1000;
                int isotopeIndex = p->getIndexByZaid(isotopeZaid);
                if(isotopeIndex >= 0)
                {
                    weightCovered += MaterialUtils::naturalAbundances[z[i]][j];
                }

                //weightCovered += 1.0;

            }
            if(weightCovered < 1E-6)
            {
                qDebug() << "No isotopes of element " << z[i] << " found in data library";
                //qDebug() << "No isotopes of element " << z[i] << "(" << qPrintable(MaterialUtils::elementNames[z[i]]) << ")" << " found in data library";
            }
            // Divide by the weight covered
        }
    }

    return true;
}

/*
int XSection::dim1() const
{
    return m_dim1;
}

int XSection::dim2() const
{
    return m_dim2;
}

int XSection::dim3() const
{
    return m_dim3;
}
*/
