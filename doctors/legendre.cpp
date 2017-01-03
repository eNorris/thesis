#include "legendre.h"

#include <cmath>
#include <limits>

double doubleFactorial(double x)
{
    if(x < 2.0)
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


