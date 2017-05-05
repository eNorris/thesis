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
    static const std::vector<int> hounsfieldRangePhantom19;
    static const std::vector<int> hounsfieldRangePhantom19Elements;
    static const std::vector<std::vector<float> > hounsfieldRangePhantom19Weights;

    // Water groups
    static const std::vector<int> water;
    static const std::vector<int> waterElements;
    static const std::vector<std::vector<float> > waterWeights;

    static bool validate();

    static std::vector<float> weightFracToAtomFrac(std::vector<int> elements, std::vector<float> weights);
    static float atomsPerGram(std::vector<int> elements, std::vector<float> atomFractions);

    const static int HOUNSFIELD19 = 1;
    const static int WATER = 2;
};

#endif // MATERIALUTILS_H
