#ifndef XSECTION_H
#define XSECTION_H

#include <vector>

//#include "config.h"

class AmpxParser;


class XSection
{
public:

    std::vector<float> m_scat2d;

    std::vector<float> m_tot1d;
    std::vector<float> m_scat1d;

    XSection();
    //XSection(const Config *config);
    ~XSection();


    //int operator()(int grp, int d2, int d3, int d4) const;

    //void load(const Config *config);

    unsigned int groupCount() const;

    float scatXs1d(const int zid, const int Esrc, const int Etar) const;
    float totXs1d(const int zid, const int E) const;

    bool allocateMemory(const unsigned int materialCount, const unsigned int groupCount, const unsigned int PnCount);
    bool addMaterial(const std::vector<int> &z, const std::vector<float> &w, const AmpxParser *p);

private:
    int m_groups;
    //int m_zids;
    int m_mats;
    int m_pns;

    //int m_dim1;
    //int m_dim2;
    //int m_dim3;
    int m_matsLoaded;
};

#endif // XSECTION_H
