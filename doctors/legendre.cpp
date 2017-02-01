#define _USE_MATH_DEFINES
#include <cmath>

#include "legendre.h"
#include <limits>
#include <QDebug>
#include <QErrorMessage>

#include "quadrature.h"

double factorial(double x)
{
    if(x <= 1.001)
        return 1.0;

    return x * factorial(x - 1.0);
}

double fastFactorial(int x)
{
    const std::vector<double> facts = {1.0, 1.0, 2.0, 6.0, 24.0, 120.0,
                                      720.0, 5040.0, 40320.0, 362880.0, 3628800.0,
                                      39916800.0, };

    return facts[x];
}

double doubleFactorial(double x)
{
    if(x < 2.001)
        return 1.0;

    return x * doubleFactorial(x-2.0);
}

Legendre::Legendre() : m_precomputed(false), m_angles(0), m_pn(0), m_table()
{

}

Legendre::~Legendre()
{

}

SOL_T Legendre::operator()(const unsigned int l, const SOL_T mu)
{
    double result;
    switch(l)
    {
    case 0:
        result = 1.0;
        break;
    case 1:
        result = mu;
        break;
    case 2:
        result = 0.5 * (3 * pow(mu, 2.0) - 1);
        break;
    case 3:
        result = 0.5 * (5 * pow(mu, 3.0) - 3*mu);
        break;
    case 4:
        result = 0.125 * (35 * pow(mu, 4.0) - 30 * pow(mu, 2.0) + 3);
        break;
    case 5:
        result = 0.125 * (63 * pow(mu, 5.0) - 70 * pow(mu, 3.0) + 15*mu);
        break;
    case 6:
        result = 0.0625 * (231 * pow(mu, 6.0) - 315 * pow(mu, 4.0) + 105*pow(mu, 2.0) - 5);
        break;
    case 7:
        result = 0.0625 * (429 * pow(mu, 7.0) - 693 * pow(mu, 5.0) + 315*pow(mu, 3.0) - 35*mu);
        break;
    case 8:
        result = 0.0078125 * (6435 * pow(mu, 8.0) - 12012 * pow(mu, 6.0) + 6930*pow(mu, 4.0) - 1260*pow(mu, 2.0) + 35);
        break;
    case 9:
        result = 0.0078125 * (12155 * pow(mu, 9.0) - 25740 * pow(mu, 7.0) + 18018*pow(mu, 5.0) - 4620*pow(mu, 3.0) + 315*mu);
        break;
    case 10:
        result = 0.00390625 * (46189 * pow(mu, 10.0) - 109395 * pow(mu, 8.0) + 90090*pow(mu, 6.0) - 30030*pow(mu, 4.0) + 3465*pow(mu, 2.0) - 63);
        break;
    default:
        qWarning() << "Can't compute Legendre polynomials beyond order 10, requested order " << l;
    };

    return static_cast<SOL_T>(result);

}

SOL_T Legendre::table(const unsigned int ia1, const unsigned int ia2, const unsigned int il)
{
    if(!m_precomputed)
    {
        qCritical() << "Attempted to access a data table before it was computed!";
        return static_cast<SOL_T>(1.0);
    }

    if(ia1 >= m_angles || ia2 >= m_angles)
    {
        qCritical() << "Attempted to access an angle that doesn't exist!  ia1=" << ia1 << "   ia2=" << ia2;
        return static_cast<SOL_T>(1.0);
    }

    if(il > m_pn)
    {
        qCritical() << "Attempted to access a Legendre coeff that doesn't exist!  il=" << il;
        return static_cast<SOL_T>(1.0);
    }

    return m_table[ia1*m_ia1jmp + ia2*m_ia2jmp + il];
}

void Legendre::precompute(const Quadrature *quad, const unsigned int pn)
{
    if(m_precomputed)
    {
        qWarning() << "Computing a table after it has already been computed!";
        m_table.clear();
    }

    m_angles = quad->angleCount();
    m_pn = pn;
    m_table.resize(m_angles * m_angles * (m_pn + 1));

    m_ia1jmp = m_angles * (m_pn + 1);
    m_ia2jmp = m_pn + 1;

    for(unsigned int ia1 = 0; ia1 < m_angles; ia1++)
        for(unsigned int ia2 = 0; ia2 < m_angles; ia2++)
            for(unsigned int il = 0; il <= m_pn; il++)
                m_table[ia1*m_ia1jmp + ia2*m_ia2jmp + il] = (*this)(il, quad->mu[ia1]*quad->mu[ia2] + quad->eta[ia1]*quad->eta[ia2] + quad->zi[ia1]*quad->zi[ia2]);

    m_precomputed = true;
}

AssocLegendre::AssocLegendre()
{

}

AssocLegendre::~AssocLegendre()
{

}


float AssocLegendre::operator()(const int l, const int m, const float x)
{
    // P_m^m(x) = (-1)^m (2m-1)!! (1-x^2)^(m/2)
    // x!! is the double factorial function
    float pmm = ((m%2==0)? (1.0) : (-1.0)) * doubleFactorial(2*m-1) * pow((1-x*x), m/2.0);
    if(l == m)
        return pmm;

    float pmm1 = x*(2*m+1)*pmm;
    if(l==m+1)
        return pmm1;

    float pml;
    int curl = m+2;  // current l in P_l^m being evaluated

    while(curl <= l)
    {
        pml = (x*(2*curl-1)*pmm1 - (curl+m-1)*pmm)/(curl-m);
        pmm = pmm1;
        pmm1 = pml;
        curl++;
    }
    return pml;
}


SphericalHarmonic::SphericalHarmonic()
{

}

SphericalHarmonic::~SphericalHarmonic()
{

}

float SphericalHarmonic::normConst(const int l, const int m)
{
    if(m == 0)
    {
        qDebug() << "Got a m=0 case in the normConst which shouldn't handle it";
    }

    //float sgn = (m%2 == 0 ? 1.0f : -1.0f);
    float t1 = static_cast<float>(2*l+1) / (2.0 * M_PI);
    float t2 = factorial(l-fabs(m)) / factorial(l + fabs(m));
    //if(m == 0)
    //{
    //    t1 /= 2.0f;
    //}
    return sqrt(t1 * t2);
}
/*
float SphericalHarmonic::operator()(const int l, const int m, const float theta, const float phi)
{
    // Y_l^m(\theta, \phi)
    if(m > 0)
    {
        return sqrt(2.0) * normConst(l, m) * cos(m * phi) * m_assoc(l, m, cos(theta));
    }
    else if(m == 0)
    {
        return normConst(l, 0) * m_assoc(l, 0, cos(theta));
    }
    else
    {
        return sqrt(2.0) * normConst(l, m) * sin(-m*phi) * m_assoc(l, -m, cos(theta));
    }
}
*/

float SphericalHarmonic::ylm_e(const int l, const int m, const float theta, const float phi)
{
    if(m <= 0)
    {
        qDebug() << "Got a neg. m value!";
    }
    return normConst(l, m) * m_assoc(l, m, cos(theta)) * cos(m * phi);
}

float SphericalHarmonic::ylm_o(const int l, const int m, const float theta, const float phi)
{
    if(m <= 0)
    {
        qDebug() << "Got a neg m value!";
    }
    return normConst(l, m) * m_assoc(l, m, cos(theta)) * sin(m * phi);
}

float SphericalHarmonic::yl0(const int l, const float theta, const float phi)
{
    return sqrt((2*l + 1)/(4.0 * M_PI)) * m_assoc(l, 0, cos(theta));
}



