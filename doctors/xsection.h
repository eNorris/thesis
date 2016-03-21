#ifndef XSECTION_H
#define XSECTION_H

#include <vector>

#include "config.h"


class XSection
{
public:

    std::vector<float> msig;

    XSection();
    XSection(const Config *config);
    ~XSection();


    int operator()(int grp, int d2, int d3, int d4) const;

    void load(const Config *config);

    int groupCount() const;

    int dim1() const;
    int dim2() const;
    int dim3() const;

private:
    int m_groups;

    int m_dim1;
    int m_dim2;
    int m_dim3;
};

#endif // XSECTION_H
