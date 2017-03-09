#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QWaitCondition>
#include <QMutex>
#include <QFileDialog>
#include <QThread>

#include "xs_reader/ampxparser.h"
#include "globals.h"

class OutputDialog;
class GeomDialog;
class QuadDialog;
class XSectionDialog;
class EnergyDialog;
class Solver;

//#include "config.h"
class Quadrature;
class Mesh;
class XSection;
class SourceParams;
//class Config;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    const static unsigned int ISOTROPIC = 0;
    const static unsigned int LEGENDRE = 1;
    const static unsigned int HARMONIC = 2;

private:
    Ui::MainWindow *ui;

    //Config *m_config;
    Mesh *m_mesh;
    XSection *m_xs;
    Quadrature *m_quad;
    SourceParams *m_params;

    OutputDialog *outputDialog;
    GeomDialog *geomDialog;
    QuadDialog *quadDialog;
    XSectionDialog *xsDialog;
    EnergyDialog *energyDialog;

    bool m_geomLoaded, m_xsLoaded, m_quadLoaded, m_paramsLoaded;

    static QPalette *m_goodPalette, *m_badPalette;

    QFileDialog *m_configSelectDialog;

    //QWaitCondition m_pendingUserContinue;
    //QMutex m_mutex;

    AmpxParser *m_parser;
    QThread m_xsWorkerThread;

    Solver *m_solver;
    QThread m_solverWorkerThread;

    std::vector<SOL_T> *m_solution;
    std::vector<RAY_T> *m_raytrace;

    unsigned int m_solType;
    unsigned int m_pn;

    void launchXsReader();

public:
    QMutex &getBlockingMutex();

protected slots:
    //void userDebugNext();
    //void userDebugAbort();
    void on_geometryOpenPushButton_clicked();
    void on_quadTypeComboBox_activated(int);
    void on_quadData1ComboBox_activated(int);
    void on_quadData2ComboBox_activated(int);
    void on_paramsTypeComboBox_activated(int);
    void on_paramsPnComboBox_activated(int);
    void on_launchSolverPushButton_clicked();
    void on_xsOpenPushButton_clicked();
    void on_xsExplorePushButton_clicked();
    void on_actionMCNP6_Generation_triggered();


    void updateLaunchButton();
    void xsParseErrorHandler(QString);
    void xsParseUpdateHandler(int x);
    void xsParseFinished(AmpxParser *parser);

    bool buildMaterials(AmpxParser *parser);

    void onRaytracerFinished(std::vector<RAY_T>* uncollided);
    void onSolverFinished(std::vector<SOL_T>* solution);

signals:
    void signalLaunchRaytracerIso(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SourceParams *params);
    void signalLaunchSolverIso(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const std::vector<RAY_T> *uflux, const SourceParams *params);

    void signalLaunchRaytracerLegendre(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const unsigned int pn, const SourceParams *params);
    void signalLaunchSolverLegendre(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const unsigned int pn, const std::vector<RAY_T> *uflux, const SourceParams *params);

    void signalLaunchRaytracerHarmonic(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const unsigned int pn, const SourceParams *params);
    void signalLaunchSolverHarmonic(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const unsigned int pn, const std::vector<RAY_T> *uflux, const SourceParams *params);

    void signalDebugHalt(std::vector<float>);
    void signalBeginXsParse(QString);

};

#endif // MAINWINDOW_H
