#ifndef XSECTION_H
#define XSECTION_H

/**
  This class uses data from the AMPX cross sections.
  */

#include <vector>

class AmpxParser;

class XSection
{
public:

    std::vector<float> m_scat2d;

    std::vector<float> m_tot1d;
    std::vector<float> m_scat1d;

    XSection();
    ~XSection();

    unsigned int groupCount() const;

    float scatXs1d(const int matid, const int g) const;
    float totXs1d(const int matid, const int g) const;
    float scatxs2d(const int matid, const int gSource, const int gSink, const int n) const;

    bool allocateMemory(const unsigned int materialCount, const unsigned int groupCount, const unsigned int PnCount);
    bool addMaterial(const std::vector<int> &z, const std::vector<float> &w, const AmpxParser *p);

private:
    int m_groups;
    int m_mats;
    int m_pn;
    int m_matsLoaded;

    const static unsigned int MT_GAMMA_TOTAL_INTERACTION = 501;
    const static unsigned int MT_GAMMA_COHERENT_SCATTER = 502;
    const static unsigned int MT_GAMMA_INCOHERENT_SCATTER = 504;
    const static unsigned int MT_GAMMA_PAIR_PRODUCTION = 516;
    const static unsigned int MT_GAMMA_PHOTOELEC_ABSORPTION = 516;
};

#endif // XSECTION_H
