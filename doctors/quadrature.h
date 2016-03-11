#ifndef QUADRATURE_H
#define QUADRATURE_H

#include <vector>

#include "config.h"

class Quadrature
{
public:
    Quadrature();
    Quadrature(const Config *config);
    Quadrature(const int sn);

    static const Quadrature& getSn2();

    std::vector<float> wt;
    std::vector<float> mu;
    std::vector<float> eta;
    std::vector<float> zi;

    void load(const Config *config);
    void load(const int sn);

private:
    static const Quadrature ms_sn2;

};

#endif // QUADRATURE_H
