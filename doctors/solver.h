#ifndef SOLVER_H
#define SOLVER_H

//#define _USE_MATH_DEFINES
#include <cmath>

#include <QObject>
#include <vector>

#include "globals.h"

class Quadrature;
class Mesh;
class XSection;
class SolverParams;
class SourceParams;

class Solver : public QObject
{
    Q_OBJECT
public:
    explicit Solver(QObject *parent = 0);
    ~Solver();

    const SOL_T m_pi = static_cast<SOL_T>(3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679);
    const SOL_T m_4pi = static_cast<SOL_T>(4.0 * m_pi);
    const SOL_T m_4pi_inv = static_cast<SOL_T>(1.0 / m_4pi);

protected:
    std::vector<RAY_T> *basicRaytrace(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SourceParams *params);

signals:
    void signalNewRaytracerIteration(std::vector<RAY_T>*);
    void signalNewSolverIteration(std::vector<SOL_T>*);

    void signalRaytracerFinished(std::vector<RAY_T>*);
    void signalSolverFinished(std::vector<SOL_T>*);

public slots:

    // Base launcher
    void raytraceIso(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar);
    void gsSolverIso(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar, const std::vector<RAY_T> *uflux);

    void raytraceLegendre(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar);
    void gsSolverLegendre(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar, const std::vector<RAY_T> *uflux);

    void raytraceHarmonic(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar);
    void gsSolverHarmonic(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar, const std::vector<RAY_T> *uflux);

    // CPU versions
    void raytraceIsoCPU(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar);
    void gsSolverIsoCPU(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar, const std::vector<RAY_T> *uflux);

    void raytraceLegendreCPU(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar);
    void gsSolverLegendreCPU(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar, const std::vector<RAY_T> *uflux);

    void raytraceHarmonicCPU(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar);
    void gsSolverHarmonicCPU(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar, const std::vector<RAY_T> *uflux);

    // The GPU versions are implemented in cuda_link which is compiled by nvcc

};

#endif // SOLVER_H
