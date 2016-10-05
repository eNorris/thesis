#include "xsection.h"

#include <cmath>

#include <QDebug>

#include "xs_reader/ampxparser.h"

XSection::XSection() : m_groups(0)
{

}

XSection::XSection(const Config *config) : m_groups(0)
{
    load(config);
}

XSection::~XSection()
{

}

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
        msig.resize(m_groups * m_dim1 * m_dim2 * m_dim3);
        xsTot.resize(m_groups*m_zids);
        xsScat.resize(m_groups*m_groups*m_zids);

        //for(int i = 0; i < m_groups; i++)
        //    for(int j = 0; j < t1; j++)
        //        for(int k = 0; k < t2; k++)
        //            for(int m = 0; m < t3; m++)
        //                msig[i*t1*t2*t3 + j*t2*t3 + k*t3 + m] = config->xsection[((j-1) * (1+config->isct) + k)*4 + config->iht + m - 1];

        for(int i = 0; i < m_groups; i++)
            for(int j = 0; j < m_dim1; j++)
                for(int k = 0; k < m_dim2; k++)
                    for(int m = 0; m < m_dim3; m++)
                        msig[i*m_dim1 + j*m_dim2 + k*m_dim3 + m] = config->xsection[((j-1) * (1+config->isct) + k)*4 + config->iht + m - 1];

        for(int zid = 0; zid < m_zids; zid++)
            for(int Esrc = 0; Esrc < m_groups; Esrc++)
                for(int Etar = 0; Etar < m_groups; Etar++)
                    xsScat[zid*m_groups*m_groups + Esrc*m_groups + Etar] = config->xsScat[zid*m_groups*m_groups + Esrc*m_groups + Etar];

        for(int zid = 0; zid < m_zids; zid++)
            for(int E = 0; E < m_groups; E++)
                xsTot[zid*m_groups + E] = config->xsTot[zid*m_groups + E];
    }
}

int XSection::operator()(int grp, int d2, int d3, int d4) const
{
    unsigned int indx = grp*m_dim1 + d2*m_dim2 + d3*m_dim3 + d4;
    if(indx >= msig.size())
        qDebug() << "XSection indexing error: Accessed " << (indx+1) << "/" << msig.size();
    return msig[grp*m_dim1 + d2*m_dim2 + d3*m_dim3 + d4];
}

unsigned int XSection::groupCount() const
{
    return m_groups;
}

float XSection::scatXs(const int zid, const int Esrc, const int Etar) const
{
    return xsScat[zid*m_groups*m_groups + Esrc*m_groups + Etar];
}

float XSection::totXs(const int zid, const int E) const
{
    return xsTot[zid*m_groups + E];
}

bool XSection::allocateMemory(const unsigned int materialCount, const unsigned int groupCount, const unsigned int PnCount)
{

}

bool XSection::addMaterial(const std::vector<int> &z, const std::vector<float> &w, const AmpxParser *p)
{

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
