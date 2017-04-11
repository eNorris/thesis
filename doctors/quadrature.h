#ifndef QUADRATURE_H
#define QUADRATURE_H

#include <vector>

//#include "config.h"

class Quadrature
{
public:
    Quadrature();
    Quadrature(const int sn);

    std::vector<float> wt;
    std::vector<float> mu;
    std::vector<float> eta;
    std::vector<float> zi;

    void loadSn(const int sn);
    void loadSpecial(const int sp);
    unsigned int angleCount() const;
    void sortIntoOctants();

private:
    unsigned int m_angles;
};

#endif // QUADRATURE_H
