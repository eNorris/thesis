#ifndef XSECTION_H
#define XSECTION_H

#include <vector>

#include "config.h"


class XSection
{
public:

    std::vector<float> msig;

    XSection();
    XSection(const Config &config);
    ~XSection();

    void load(const Config &config);
};

#endif // XSECTION_H
