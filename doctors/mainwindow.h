#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

class OutputDialog;
class GeomDialog;
class QuadDialog;
class XSectionDialog;

//#include "config.h"
#include "quadrature.h"
#include "mesh.h"
#include "xsection.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

    Config *config;

    OutputDialog *outputDialog;
    GeomDialog *geomDialog;
    QuadDialog *quadDialog;
    XSectionDialog *xsDialog;

    std::vector<float> gssolver(const Quadrature *quad, const Mesh *mesh, const XSection *xs);

signals:
    void signalNewIteration(std::vector<float>);
};

#endif // MAINWINDOW_H
