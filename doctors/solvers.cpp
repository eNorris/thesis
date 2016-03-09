//#include "solvers.h"
#include "mainwindow.h"

std::vector<float> MainWindow::gssolver(const Quadrature &quad, const Mesh &mesh, const XSection &xs)
{
    std::vector<float> scalarFlux;
    std::vector<float> angularFlux;

    for(int i = 0; i < 5; i++)
    {
        // Do the solving...

        emit signalNewIteration(scalarFlux);
    }

    return scalarFlux;
}

