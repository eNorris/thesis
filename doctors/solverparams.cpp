#include "solverparams.h"

SolverParams::SolverParams()
{

}

SolverParams::~SolverParams()
{

}

void SolverParams::normalize()
{
    float t = 0;
    for(unsigned int i = 0; i < spectraIntensity.size(); i++)
        t += spectraIntensity[i];
    for(unsigned int i = 0; i < spectraIntensity.size(); i++)
        spectraIntensity[i] /= t;
    return;
}
