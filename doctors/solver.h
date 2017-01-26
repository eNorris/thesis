#ifndef SOLVER_H
#define SOLVER_H

#define _USE_MATH_DEFINES
#include <cmath>

#include <QObject>
#include <vector>

class Quadrature;
class Mesh;
class XSection;

class Solver : public QObject
{
    Q_OBJECT
public:
    explicit Solver(QObject *parent = 0);
    ~Solver();

    const float M_4PI = 4 * M_PI;

signals:
    void signalNewIteration(std::vector<float>*);
    //void signalRaytracerIsoFinished(std::vector<float>*);
    //void signalSolverIsoFinished(std::vector<float>*);

    void signalRaytracerFinished(std::vector<float>*);
    void signalSolverFinished(std::vector<float>*);

public slots:
    void raytraceIso(const Quadrature *quad, const Mesh *mesh, const XSection *xs);
    void gsSolverIso(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const std::vector<float> *uflux);

    void raytraceLegendre(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const unsigned int pn);
    void gsSolverLegendre(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const unsigned int pn, const std::vector<float> *uflux);

    void raytraceHarmonic(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const unsigned int pn);
    void gsSolverHarmonic(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const unsigned int pn, const std::vector<float> *uflux);

};

#endif // SOLVER_H
