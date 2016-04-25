#ifndef XSECTION_H
#define XSECTION_H

#include <vector>

#include "config.h"


class XSection
{
public:

    std::vector<float> msig;

    std::vector<float> xsTot;
    std::vector<float> xsScat;

    XSection();
    XSection(const Config *config);
    ~XSection();


    int operator()(int grp, int d2, int d3, int d4) const;

    void load(const Config *config);

    unsigned int groupCount() const;

    float scatXs(const int zid, const int Esrc, const int Etar) const;
    float totXs(const int zid, const int E) const;

    //int dim1() const;
    //int dim2() const;
    //int dim3() const;

private:
    int m_groups;
    int m_zids;

    int m_dim1;
    int m_dim2;
    int m_dim3;
};

#endif // XSECTION_H
