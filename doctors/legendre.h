#ifndef LEGENDRE_H
#define LEGENDRE_H

#include <iostream>
#include <vector>

double factorial(double x);
double doubleFactorial(double x);

class Polynomial : public std::vector<float>
{
public:
    Polynomial();
    ~Polynomial();

    float eval(float x);
    void setNan(bool isNan);
    //void setSqrtTermFlag(bool flag);
    //bool getSqrtTermFlag();

    Polynomial operator*(const Polynomial &other) const;

protected:
    bool m_nan;
    //bool m_sqrtTerm;

    //static Polynomial s_sqrtTerm;
};

std::ostream &operator<<(std::ostream &out, const Polynomial &p);

class Legendre
{
public:
    Legendre();
    ~Legendre();

    void expandMaxOrder(int ord);
    Polynomial getLegendrePoly(int l);
    Polynomial calculateLegendrePoly(int order);
    float legendreEval(int l, float x);

private:
    std::vector<Polynomial> m_legendreCoeff;
    int m_maxOrder;

};

class AssocLegendre
{
public:
    AssocLegendre();
    ~AssocLegendre();

    //void expandMaxOrder(int order);
    //Polynomial getAssocLegendrePoly(int l, int m);
    //Polynomial calculateAssocLegendrePoly(int order);
    float assocLegendreEval(int l, int m, float x);

protected:
    std::vector<std::vector<Polynomial> > m_assocCoeff;
    int m_maxOrder;
};



#endif // LEGENDRE_H
