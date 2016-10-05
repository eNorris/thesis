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
    //XSection *m_xs;
    std::vector<XSection *> m_mats;
    Quadrature *m_quad;

    OutputDialog *outputDialog;
    GeomDialog *geomDialog;
    QuadDialog *quadDialog;
    XSectionDialog *xsDialog;

    bool m_geomLoaded, m_xsLoaded, m_quadLoaded, m_paramsLoaded;

    static QPalette *m_goodPalette, *m_badPalette;

    QFileDialog *m_configSelectDialog;

    QWaitCondition m_pendingUserContinue;
    QMutex m_mutex;

    AmpxParser *m_parser;
    QThread m_xsWorkerThread;

    // Implemented in solvers.cpp instead of mainwindow.cpp
    std::vector<float> gssolver(const Quadrature *quad, const Mesh *mesh, const std::vector<XSection *> &mats, const Config *config, const std::vector<float> *uflux);

    // Implemented in raytracer.cpp instead of mainwindow.cpp
    std::vector<float> raytrace(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const Config *config);

    bool buildMaterials(AmpxParser *parser);
    XSection *makeMaterial(std::vector<int> z, std::vector<float> w, AmpxParser *ampxParser);

    void launchXsReader();

public:
    QMutex &getBlockingMutex();

protected slots:
    void launchSolver();
    void userDebugNext();
    void userDebugAbort();
    //void slotLoadConfigClicked();
    void slotOpenCtData();
    void on_quadTypeComboBox_activated(int);
    void on_quadData1ComboBox_activated(int);
    void on_quadData2ComboBox_activated(int);
    //void slotQuadSelected(int);
    //void slotQuadSelected(int);
    void on_xsOpenPushButton_clicked();
    void on_xsExplorePushButton_clicked();

    void updateLaunchButton();
    void xsParseErrorHandler(QString);
    void xsParseUpdateHandler(int x);

signals:
    void signalNewIteration(std::vector<float>);
    void signalDebugHalt(std::vector<float>);
    void signalBeginXsParse(QString);

};

#endif // MAINWINDOW_H
