#ifndef SOLVERPARAMS_H
#define SOLVERPARAMS_H

#include <vector>

#include <QObject>

class SolverParams : public QObject
{
    Q_OBJECT

public:
    SolverParams();
    ~SolverParams();

    float sourceX, sourceY, sourceZ;

    std::vector<float> spectraEnergyLimits;
    std::vector<float> spectraIntensity;

public slots:
    void normalize();
};

#endif // SOLVERPARAMS_H
