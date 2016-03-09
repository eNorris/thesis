#ifndef QUADRATURE_H
#define QUADRATURE_H

#include <vector>

#include "config.h"

class Quadrature
{
public:
    Quadrature();
    Quadrature(const Config &config);

    std::vector<float> wt;
    std::vector<float> mu;
    std::vector<float> eta;
    std::vector<float> zi;

    void load(const Config &config);
};

#endif // QUADRATURE_H
