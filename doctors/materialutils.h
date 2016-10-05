#ifndef MATERIALUTILS_H
#define MATERIALUTILS_H

#include <vector>

class MaterialUtils
{
public:
    MaterialUtils();
    ~MaterialUtils();

    static const std::vector<float> atomicMass;
    static const std::vector<std::vector<int> > naturalIsotopes;

    float getAtomicMass(int z);
    std::vector<float> getNatIsotopes(int a);
    std::vector<float> getNatIsotopeFracs(int a);
};

#endif // MATERIALUTILS_H
