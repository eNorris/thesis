#define _USE_MATH_DEFINES
#include <cmath>

#include "legendre.h"
#include <limits>
#include <QDebug>

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

Legendre::Legendre()
{

}

Legendre::~Legendre()
{

}

SOL_T Legendre::operator()(const int l, const SOL_T mu)
{
    double result;
    switch(l)
    {
    case 0:
        result = 1.0;
    case 1:
        result = mu;
    case 2:
        result = 0.5 * (3 * pow(mu, 2.0) - 1);
    case 3:
        result = 0.5 * (5 * pow(mu, 3.0) - 3*mu);
    case 4:
        result = 0.125 * (35 * pow(mu, 4.0) - 30 * pow(mu, 2.0) + 3);
    case 5:
        result = 0.125 * (63 * pow(mu, 5.0) - 70 * pow(mu, 3.0) + 15*mu);
    case 6:
        result = 0.0625 * (231 * pow(mu, 6.0) - 315 * pow(mu, 4.0) + 105*pow(mu, 2.0) - 5);
    case 7:
        result = 0.0625 * (429 * pow(mu, 7.0) - 693 * pow(mu, 5.0) + 315*pow(mu, 3.0) - 35*mu);
    case 8:
        result = 0.0078125 * (6435 * pow(mu, 8.0) - 12012 * pow(mu, 6.0) + 6930*pow(mu, 4.0) - 1260*pow(mu, 2.0) + 35);
    case 9:
        result = 0.0078125 * (12155 * pow(mu, 9.0) - 25740 * pow(mu, 7.0) + 18018*pow(mu, 5.0) - 4620*pow(mu, 3.0) + 315*mu);
    case 10:
        result = 0.00390625 * (46189 * pow(mu, 10.0) - 109395 * pow(mu, 8.0) + 90090*pow(mu, 6.0) - 30030*pow(mu, 4.0) + 3465*pow(mu, 2.0) - 63);
    default:
        qWarning << "Can't compute Legendre polynomials beyond order ";
    };

    return static_cast<SOL_T>(result);

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



/*
float AssocLegendre::operator ()(const int l, const int m, const float x)
{
    if(l == 0 && m == 0)
    {
        return 1.0;
    }
    else if(l == m)
    {
        return ((m%2==0)? (1.0) : (-1.0)) * doubleFactorial(2*m - 1) * pow(1-x*x, m/2.0);
    }
    else if(l == m+1)
    {
        return x * (2*m+1) * operator ()(m, m, x);
    }
    else
    {
        return x*(2*l-1)/(l-m) * operator()(l-1, m, x) - (l+m-1.0)/(l-m)*operator()(l-2, m, x);
    }
}
*/


SphericalHarmonic::SphericalHarmonic()
{

}

SphericalHarmonic::~SphericalHarmonic()
{

}

float SphericalHarmonic::normConst(const int l, const int m)
{
    //float sgn = (m%2 == 0 ? 1.0f : -1.0f);
    float t1 = static_cast<float>(2*l+1) / (2.0 * M_PI);
    float t2 = factorial(l-fabs(m)) / factorial(l + fabs(m));
    if(m == 0)
    {
        t1 /= 2.0f;
    }
    return sqrt(t1 * t2);
}

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

float SphericalHarmonic::ylm_e(const int l, const int m, const float theta, const float phi)
{
    if(m < 0)
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



