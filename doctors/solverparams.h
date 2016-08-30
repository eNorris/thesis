#ifndef SOLVERPARAMS_H
#define SOLVERPARAMS_H

#include <vector>

class SolverParams
{
public:
    SolverParams();
    ~SolverParams();

    float sourceX, sourceY, sourceZ;

    std::vector<float> spectraEnergy;
    std::vector<float> spectraIntensity;
};

#endif // SOLVERPARAMS_H
