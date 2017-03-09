#ifndef SOLVERPARAMS_H
#define SOLVERPARAMS_H

class SolverParams
{
public:
    SolverParams();
    ~SolverParams();

    unsigned int pn;
    bool gpu_accel;
};

#endif // SOLVERPARAMS_H
