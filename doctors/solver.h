#ifndef SOLVER_H
#define SOLVER_H

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

signals:
    void signalNewIteration(std::vector<float>*);
    void signalRaytracerFinished(std::vector<float>*);
    void signalSolverFinished(std::vector<float>*);

public slots:
    void raytraceIso(const Quadrature *quad, const Mesh *mesh, const XSection *xs);
    void gsSolverIso(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const std::vector<float> *uflux);

    void raytrace(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const int pn);
    void gssolver(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const int pn, const std::vector<float> *uflux);

};

#endif // SOLVER_H
