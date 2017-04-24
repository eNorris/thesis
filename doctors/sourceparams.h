#ifndef SOURCEPARAMS_H
#define SOURCEPARAMS_H

#include <vector>

class AmpxParser;

class SourceParams
{
public:
    SourceParams(AmpxParser *parser);
    ~SourceParams();

    int sourceType;

    float sourceX, sourceY, sourceZ;
    float sourcePhi, sourceTheta;
    int sourceN;
    bool degrees;

    float sourceD, sourceW, sourceH;

    std::vector<float> spectraEnergyLimits;
    std::vector<float> spectraIntensity;

    bool normalize();
    bool update(std::vector<float> e, double x, double y, double z, double phi, double theta, int n, bool degrees, double d, double w, double h, int type);
};

#endif // SOURCEPARAMS_H
