#define _USE_MATH_DEFINES
#include <cmath>

#include "legendre.h"
#include <limits>

double factorial(double x)
{
    if(x <= 1.001)
        return 1.0;

    return x * factorial(x - 1.0);
}

double doubleFactorial(double x)
{
    if(x < 2.001)
        return 1.0;

    return x * doubleFactorial(x-2.0);
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


SphericalHarmonic::SphericalHarmonic()
{

}

SphericalHarmonic::~SphericalHarmonic()
{

}

float SphericalHarmonic::normConst(const int l, const int m)
{
    float t1 = static_cast<float>(2*l+1) / (4 * M_PI);
    float t2 = factorial(l-fabs(m)) / factorial(l + fabs(m));
    return sqrt(t1 * t2);
}

float SphericalHarmonic::operator()(const int l, const int m, const float theta, const float phi)
{
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



