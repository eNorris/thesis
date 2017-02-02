#ifndef SOLVER_H
#define SOLVER_H

#define _USE_MATH_DEFINES
#include <cmath>

#include <QObject>
#include <vector>

#include "globals.h"

class Quadrature;
class Mesh;
class XSection;

class Solver : public QObject
{
    Q_OBJECT
public:
    explicit Solver(QObject *parent = 0);
    ~Solver();

    const float M_4PI = 4.0 * M_PI;
    const float M_4PI_INV = 1.0 / M_4PI;

signals:
    //void signalNewIteration(std::vector<float>*);
    void signalNewRaytracerIteration(std::vector<RAY_T>*);
    void signalNewSolverIteration(std::vector<SOL_T>*);
    //void signalNewSolverIteration(std::vector<RAY_T>*);
    //void signalRaytracerIsoFinished(std::vector<float>*);
    //void signalSolverIsoFinished(std::vector<float>*);

    void signalRaytracerFinished(std::vector<RAY_T>*);
    void signalSolverFinished(std::vector<SOL_T>*);

public slots:
    void raytraceIso(const Quadrature *quad, const Mesh *mesh, const XSection *xs);
    void gsSolverIso(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const std::vector<RAY_T> *uflux);

    void raytraceLegendre(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const unsigned int pn);
    void gsSolverLegendre(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const unsigned int pn, const std::vector<RAY_T> *uflux);

    void raytraceHarmonic(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const unsigned int pn);
    void gsSolverHarmonic(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const unsigned int pn, const std::vector<RAY_T> *uflux);

};

#endif // SOLVER_H
