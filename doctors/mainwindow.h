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
class SolverParams;
class CtDataManager;
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
    const static unsigned int KLEIN = 3;

private:
    Ui::MainWindow *ui;

    //Config *m_config;
    Mesh *m_mesh;
    XSection *m_xs;
    Quadrature *m_quad;
    SourceParams *m_srcParams;
    SolverParams *m_solvParams;

    OutputDialog *outputDialog;
    GeomDialog *geomDialog;
    QuadDialog *quadDialog;
    XSectionDialog *xsDialog;
    EnergyDialog *energyDialog;

    bool m_geomLoaded, m_xsLoaded, m_quadLoaded, m_solverLoaded, m_srcLoaded;

    static QPalette *m_goodPalette, *m_badPalette;

    QFileDialog *m_configSelectDialog;

    //QWaitCondition m_pendingUserContinue;
    //QMutex m_mutex;

    AmpxParser *m_parser;
    QThread m_xsWorkerThread;

    CtDataManager *m_ctman;
    QThread m_meshWorkerThread;

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
    void on_solverGpuCheckBox_toggled(bool);
    void on_sourceTypeComboBox_activated(int);

    void updateLaunchButton();
    void xsParseErrorHandler(QString);
    void xsParseUpdateHandler(int x);
    void xsParseFinished(AmpxParser *parser);

    void meshParseUpdateHandler(int x);
    void meshParseFinished(Mesh *mesh);

    bool buildMaterials(AmpxParser *parser);

    void energyGroupsUpdated();

    void onRaytracerFinished(std::vector<RAY_T>* uncollided);
    void onSolverFinished(std::vector<SOL_T>* solution);

signals:
    void signalLaunchRaytracerIso(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar);
    void signalLaunchSolverIso(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar, const std::vector<RAY_T> *uflux);

    void signalLaunchRaytracerLegendre(const Quadrature *quad, const Mesh *mesh, const XSection *xs,  const SolverParams *solPar, const SourceParams *srcPar);
    void signalLaunchSolverLegendre(const Quadrature *quad, const Mesh *mesh, const XSection *xs,  const SolverParams *solPar, const SourceParams *srcPar, const std::vector<RAY_T> *uflux);

    void signalLaunchRaytracerHarmonic(const Quadrature *quad, const Mesh *mesh, const XSection *xs,  const SolverParams *solPar, const SourceParams *srcPar);
    void signalLaunchSolverHarmonic(const Quadrature *quad, const Mesh *mesh, const XSection *xs,  const SolverParams *solPar, const SourceParams *srcPar, const std::vector<RAY_T> *uflux);

    void signalLaunchRaytracerKlein(const Quadrature *quad, const Mesh *mesh, const XSection *xs,  const SolverParams *solPar, const SourceParams *srcPar);
    void signalLaunchSolverKlein(const Quadrature *quad, const Mesh *mesh, const XSection *xs,  const SolverParams *solPar, const SourceParams *srcPar, const std::vector<RAY_T> *uflux);

    void signalDebugHalt(std::vector<float>);
    void signalBeginXsParse(QString);

    void signalBeginMeshParse(int,int,int,QString);

};

#endif // MAINWINDOW_H
