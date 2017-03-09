#ifndef SOURCEPARAMS_H
#define SOURCEPARAMS_H

#include <vector>

class AmpxParser;

class SourceParams
{
public:
    SourceParams(AmpxParser *parser);
    ~SourceParams();

    float sourceX, sourceY, sourceZ;

    std::vector<float> spectraEnergyLimits;
    std::vector<float> spectraIntensity;

    bool normalize();
    bool update(std::vector<float> e, double x, double y, double z);
};

#endif // SOURCEPARAMS_H
