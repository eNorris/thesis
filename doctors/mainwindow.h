#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QWaitCondition>
#include <QMutex>
#include <QFileDialog>
#include <QThread>

#include "xs_reader/ampxparser.h"

class OutputDialog;
class GeomDialog;
class QuadDialog;
class XSectionDialog;
class Solver;

//#include "config.h"
class Quadrature;
class Mesh;
class XSection;
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

private:
    Ui::MainWindow *ui;

    //Config *m_config;
    Mesh *m_mesh;
    XSection *m_xs;
    Quadrature *m_quad;

    OutputDialog *outputDialog;
    GeomDialog *geomDialog;
    QuadDialog *quadDialog;
    XSectionDialog *xsDialog;

    bool m_geomLoaded, m_xsLoaded, m_quadLoaded, m_paramsLoaded;

    static QPalette *m_goodPalette, *m_badPalette;

    QFileDialog *m_configSelectDialog;

    //QWaitCondition m_pendingUserContinue;
    //QMutex m_mutex;

    AmpxParser *m_parser;
    QThread m_xsWorkerThread;

    Solver *m_solver;
    QThread m_solverWorkerThread;

    std::vector<float> *m_solution;
    std::vector<float> *m_raytrace;

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
    void on_launchSolverPushButton_clicked();
    void on_xsOpenPushButton_clicked();
    void on_xsExplorePushButton_clicked();

    void updateLaunchButton();
    void xsParseErrorHandler(QString);
    void xsParseUpdateHandler(int x);

    bool buildMaterials(AmpxParser *parser);

    void onRaytracerFinished(std::vector<float>* uncollided);
    void onSolverFinished(std::vector<float>* solution);

signals:
    void signalLaunchRaytracer(const Quadrature *quad, const Mesh *mesh, const XSection *xs);
    void signalLaunchSolver(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const std::vector<float> *uflux);
    void signalDebugHalt(std::vector<float>);
    void signalBeginXsParse(QString);

};

#endif // MAINWINDOW_H
