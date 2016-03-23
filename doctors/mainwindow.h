#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QWaitCondition>
#include <QMutex>

class OutputDialog;
class GeomDialog;
class QuadDialog;
class XSectionDialog;

//#include "config.h"
class Quadrature;
class Mesh;
class XSection;
class Config;

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

    Config *m_config;
    Mesh *m_mesh;
    XSection *m_xs;
    Quadrature *m_quad;

    OutputDialog *outputDialog;
    GeomDialog *geomDialog;
    QuadDialog *quadDialog;
    XSectionDialog *xsDialog;

    QWaitCondition m_pendingUserContinue;
    QMutex m_mutex;

    // Implemented in solvers.cpp instead of mainwindow.cpp
    std::vector<float> gssolver(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const Config *config);

public:
    QMutex &getBlockingMutex();

protected slots:
    void launchSolver();
    void userDebugNext();
    void userDebugAbort();

signals:
    void signalNewIteration(std::vector<float>);
    void signalDebugHalt(std::vector<float>);
};

#endif // MAINWINDOW_H
