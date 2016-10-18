#ifndef MATERIALUTILS_H
#define MATERIALUTILS_H

#include <vector>
#include <string>

class MaterialUtils
{
public:

    const static float AVOGADRO;

    MaterialUtils();
    ~MaterialUtils();

    static const std::vector<std::string> elementNames;
    static const std::vector<float> atomicMass;
    static const std::vector<std::vector<int> > naturalIsotopes;
    static const std::vector<std::vector<float> > naturalAbundances;

    // Human phantom 19 groups
    const static std::vector<int> hounsfieldRangePhantom19;
    const static std::vector<int> hounsfieldRangePhantom19Elements;
    const static std::vector<std::vector<float> > hounsfieldRangePhantom19Weights;

    static bool validate();

    static std::vector<float> weightFracToAtomFrac(std::vector<int> elements, std::vector<float> weights);
    static float atomsPerGram(std::vector<int> elements, std::vector<float> atomFractions);
};

#endif // MATERIALUTILS_H
