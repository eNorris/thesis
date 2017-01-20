#ifndef LEGENDRE_H
#define LEGENDRE_H

#include <iostream>
#include <vector>

double doubleFactorial(double x);


class AssocLegendre
{
public:
    AssocLegendre();
    ~AssocLegendre();

    float operator()(const int l, const int m, const float x);
};

class SphericalHarmonic
{
public:
    SphericalHarmonic();
    ~SphericalHarmonic();

    float normConst(const int l, const int m);
    float operator()(const int l, const int m, const float theta, const float phi);
    float ylm_e(const int l, const int m, const float theta, const float phi);
    float ylm_o(const int l, const int m, const float theta, const float phi);

protected:
    AssocLegendre m_assoc;
};



#endif // LEGENDRE_H
