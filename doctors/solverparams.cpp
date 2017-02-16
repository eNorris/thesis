#include "solverparams.h"

#include "xs_reader/ampxparser.h"

SolverParams::SolverParams(AmpxParser *parser)
{
    //spectraEnergyLimits.resize(parser->getGammaEnergyGroups()+1);
    spectraIntensity.resize(parser->getGammaEnergyGroups(), 0);

    spectraEnergyLimits = parser->getGammaEnergy();
    //for(unsigned int i = 0; i < parser->getGammaEnergyGroups()+1; i++)
    //    spectra
}

SolverParams::~SolverParams()
{

}

bool SolverParams::normalize()
{
    float t = 0;

    for(unsigned int i = 0; i < spectraIntensity.size(); i++)
        if(spectraIntensity[i] < 0)
            spectraIntensity[i] = 0;

    for(unsigned int i = 0; i < spectraIntensity.size(); i++)
        t += spectraIntensity[i];

    if(t <= 1e-6)
    {
        //QString errmsg = QString("The total energy spectra is zero.");
        //QMessageBox::warning(this, "Divide by zero", errmsg, QMessageBox::Close);
        return false;
    }

    for(unsigned int i = 0; i < spectraIntensity.size(); i++)
        spectraIntensity[i] /= t;
    return true;
}

bool SolverParams::update(std::vector<float> e, double x, double y, double z)
{
    if(spectraIntensity.size() != e.size())
    {
        return false;
    }
    spectraIntensity = e;
    sourceX = x;
    sourceY = y;
    sourceZ = z;
    return true;
}
