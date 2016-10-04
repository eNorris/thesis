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



#endif // LEGENDRE_H
