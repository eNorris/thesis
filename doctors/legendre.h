#ifndef LEGENDRE_H
#define LEGENDRE_H

#include "globals.h"

#include <iostream>
#include <vector>

class Quadrature;

double doubleFactorial(double x);

class Legendre
{
public:
    Legendre();
    ~Legendre();

    SOL_T operator()(const unsigned int l, const SOL_T mu);
    SOL_T table(const unsigned int ia1, const unsigned int ia2, const unsigned int il);
    void precompute(const Quadrature *quad, const unsigned int pn);

protected:
    bool m_precomputed;
    unsigned int m_angles;
    unsigned int m_pn;
    unsigned int m_ia1jmp;
    unsigned int m_ia2jmp;
    std::vector<SOL_T> m_table;
};

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
    //float operator()(const int l, const int m, const float theta, const float phi);
    float ylm_e(const int l, const int m, const float theta, const float phi);
    float ylm_o(const int l, const int m, const float theta, const float phi);
    float yl0(const int l, const float theta, const float phi);

protected:
    AssocLegendre m_assoc;
};



#endif // LEGENDRE_H
