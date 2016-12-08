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
    void raytracerFinished(std::vector<float>*);
    void solverFinished(std::vector<float>*);

public slots:

    //std::vector<float> gssolver(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const std::vector<float> *uflux);
    //std::vector<float> raytrace(const Quadrature *quad, const Mesh *mesh, const XSection *xs);

    void gssolver(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const std::vector<float> *uflux);
    void raytrace(const Quadrature *quad, const Mesh *mesh, const XSection *xs);
};

#endif // SOLVER_H
