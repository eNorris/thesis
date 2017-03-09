#ifndef SOLVERPARAMS_H
#define SOLVERPARAMS_H

#include <vector>

//#include <QObject>

class AmpxParser;

class SourceParams
{
public:
    SourceParams(AmpxParser *parser);
    //SolverParams(std::vector<float> e, double x, double y, double z);
    ~SourceParams();

    float sourceX, sourceY, sourceZ;

    std::vector<float> spectraEnergyLimits;
    std::vector<float> spectraIntensity;

    bool normalize();
    bool update(std::vector<float> e, double x, double y, double z);
};

#endif // SOLVERPARAMS_H
