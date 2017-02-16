#ifndef SOLVERPARAMS_H
#define SOLVERPARAMS_H

#include <vector>

#include <QObject>

class AmpxParser;

class SolverParams : public QObject
{
    Q_OBJECT

public:
    SolverParams(AmpxParser *parser);
    //SolverParams(std::vector<float> e, double x, double y, double z);
    ~SolverParams();

    float sourceX, sourceY, sourceZ;

    std::vector<float> spectraEnergyLimits;
    std::vector<float> spectraIntensity;

public slots:
    bool normalize();
    bool update(std::vector<float> e, double x, double y, double z);
};

#endif // SOLVERPARAMS_H
