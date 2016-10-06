#ifndef MATERIALUTILS_H
#define MATERIALUTILS_H

#include <vector>
#include <string>

class MaterialUtils
{
public:
    MaterialUtils();
    ~MaterialUtils();

    static const std::vector<std::string> elementNames;
    static const std::vector<float> atomicMass;
    static const std::vector<std::vector<int> > naturalIsotopes;
    static const std::vector<std::vector<float> > naturalAbundances;

    static bool validate();
};

#endif // MATERIALUTILS_H
