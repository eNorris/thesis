#include "legendre.h"

#include <cmath>
#include <limits>

/*
Polynomial Polynomial::s_sqrtTerm;
{
Polynomial::s_sqrtTerm.push_back(1);
}
*/

double factorial(double x)
{
    if(x < 2.0)
        return 1.0;
    return x * factorial(x-1.0);
}

double doubleFactorial(double x)
{
    if(x < 2.0)
        return 1.0;

    //if((int)(x) % 2 == 0)
    //{
    return x * doubleFactorial(x-2.0);
    //} else {
    //
    //}
}

//
// Polynomial Class
//
Polynomial::Polynomial() : m_nan(false)
{

}

Polynomial::~Polynomial()
{

}

void Polynomial::setNan(bool isNan)
{
    m_nan = isNan;
}

float Polynomial::eval(float x)
{
    if(m_nan || size() == 0)
        return std::numeric_limits<float>::signaling_NaN();

    float s = operator[](0);

    for(int i = 1; i < size(); i++)
        s += operator[](i) * pow(x, i);

    return s;
}

Polynomial Polynomial::operator*(const Polynomial &other) const
{
    Polynomial r;

    r.resize(size() + other.size() - 1, 0.0f);

    for(int i = 0; i < size(); i++)
        for(int j = 0; j < other.size(); j++)
            r[i+j] += operator[](i) * other[j];

    return r;
}

/*
void Polynomial::setSqrtTermFlag(bool flag)
{
    m_sqrtTerm = flag;
}

bool Polynomial::getSqrtTermFlag() const
{
    return m_sqrtTerm;
}
*/

std::ostream &operator<<(std::ostream &out, const Polynomial &p)
{
    bool isZero = true;
    for(int i = p.size()-1; i >= 0; i--)
        if(p[i] != 0 && i != 0)
        {
            if(i < p.size()-1)
                out << " + ";
            out << p[i] << "x^" << i;
            isZero = false;
        }
        else if(i == 0)
        {
            if(isZero)
                out << p[0];
            else if(p[0] != 0)
                out << p[0];
        }
    return out;
}

//
// Legendre Class
//

Legendre::Legendre() : m_maxOrder(-1)
{

}

Legendre::~Legendre()
{

}

void Legendre::expandMaxOrder(int ord)
{
    if(m_maxOrder >= ord)
    {
        return;
    }

    while(m_maxOrder < ord)
    {
        m_maxOrder++;
        Polynomial p = calculateLegendrePoly(m_maxOrder);
        m_legendreCoeff.push_back(p);
    }
}

Polynomial Legendre::calculateLegendrePoly(int order)
{
    int M;
    if(order % 2 == 0)
        M = order/2;
    else
        M = (order-1)/2;

    Polynomial p;
    p.resize(order+1, 0.0);

    for(int m = 0; m <= M; m++)
    {
        double sgn = (m % 2 == 0 ? 1.0 : -1.0);

        double numer = factorial(2.0*(order - m));

        int denom = pow(2.0, order) * factorial(m) * factorial(order - m) * factorial(order - 2*m);

        p[order - 2*m] = sgn * numer/denom;
    }

    return p;
}



Polynomial Legendre::getLegendrePoly(int l)
{
    if(l > m_maxOrder)
        expandMaxOrder(l);
    return m_legendreCoeff[l];
}


float Legendre::legendreEval(int l, float x)
{
    //if(l > m_maxOrder)
    //    expandMaxOrder(l);

    getLegendrePoly(l).eval(x);
}


//
// Associated Legendre Class
//

AssocLegendre::AssocLegendre() //: m_maxOrder(-1)
{

}

AssocLegendre::~AssocLegendre()
{

}
/*
void AssocLegendre::expandMaxOrder(int order)
{
    if(m_maxOrder >= order)
    {
        return;
    }

    while(m_maxOrder < order)
    {
        m_maxOrder++;
        calculateAssocLegendrePoly(m_maxOrder);
        //m_legendreCoeff.push_back(p);
    }
}

Polynomial AssocLegendre::getAssocLegendrePoly(int l, int m)
{
    if(m > l)
    {
        Polynomial p;
        p.setNan(true);
        return p;
    }
    if(l > m_maxOrder)
        expandMaxOrder(l);

    return m_legendreCoeff[l][m];
}

Polynomial AssocLegendre::calculateAssocLegendrePoly(int order)
{
    std::vector<Polynomial> v;
    v.resize(order);

    // P_l^l = (-1)^m (2m-1)!! (1-x^2)^(m/2)
    // TODO

    // P_l^m-1 = x(2m+1)P_l^l(x)

    for(int m = 0; m <= order; m++)
    {
        Polynomial p;

        v.push_back(p);
    }

    m_assocCoeff.push_back(v);
}
*/
float AssocLegendre::assocLegendreEval(int l, int m, float x)
{
    float pmm = ((m%2==0)? (1.0) : (-1.0)) * doubleFactorial(2*m-1) * pow((1-x*x), m/2);
    if(l == m)
        return pmm;

    float pmm1 = x*(2*m+1)*pmm;
    if(l==m+1)
        return pmm1;

    float pml;
    int lindx = m;

    while(l > m)
    {
        pml = (x*(2*l-1)*pmm1 - (l+m-1)*pmm)/(l-m);
        pmm1 = pml;
        pmm = pmm1;
    }
    return pml;

    //return getAssocLegendrePoly(l, m).eval(x);
}


