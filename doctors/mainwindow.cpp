#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QDebug>
#include <QFileDialog>
#include <QDir>

#include "quadrature.h"
#include "mesh.h"
#include "xsection.h"

#include "outputdialog.h"
#include "geomdialog.h"
#include "quaddialog.h"
#include "xsectiondialog.h"
#include "outwriter.h"
#include "ctdatamanager.h"

#include "config.h"
//#include "solvers.h"

MainWindow::MainWindow(QWidget *parent):
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    m_config(NULL),
    m_mesh(NULL),
    m_xs(NULL),
    m_quad(NULL),
    m_configSelectDialog(NULL)
{
    ui->setupUi(this);

    outputDialog = new OutputDialog(this);
    geomDialog = new GeomDialog(this);
    quadDialog = new QuadDialog(this);
    xsDialog = new XSectionDialog(this);

    m_configSelectDialog = new QFileDialog(this);
    m_configSelectDialog->setAcceptMode(QFileDialog::AcceptOpen);
    m_configSelectDialog->setFileMode(QFileDialog::ExistingFile);

    connect(ui->actionSolution_Explorer, SIGNAL(triggered()), outputDialog, SLOT(show()));

    connect(ui->selectQuadPushButton, SIGNAL(clicked()), quadDialog, SLOT(show()));

    connect(ui->buildGeomPushButton, SIGNAL(clicked()), geomDialog, SLOT(show()));

    connect(ui->loadXsPushButton, SIGNAL(clicked()), xsDialog, SLOT(show()));

    connect(ui->launchSolverPushButton, SIGNAL(clicked()), this, SLOT(launchSolver()));

    connect(ui->loadInputPushButton, SIGNAL(clicked()), this, SLOT(slotLoadConfigClicked()));

    //connect(this, SIGNAL(signalNewIteration(std::vector<float>)), outputDialog, SLOT(disp(std::vector<float>)));
    // TODO - Update like above at some point
    connect(this, SIGNAL(signalNewIteration(std::vector<float>)), outputDialog, SLOT(reRender(std::vector<float>)));



    // Make a configuration object and load its defaults
    //config = new Config;
    m_config = new Config;
    //m_config->loadFile("testinput.cfg");
    m_config->loadDefaults();

    qDebug() << "Loaded default configuration";

    //Quadrature *m_quad = new Quadrature(config);
    m_quad = new Quadrature(m_config);
    quadDialog->updateQuad(m_quad);

    //Mesh *mesh = new Mesh(config, quad);
    //m_mesh = new Mesh(m_config, m_quad);
    CtDataManager datareader;
    m_mesh = datareader.parse(256, 256, 64, 16, "/media/Storage/thesis/doctors/liver_volume.bin", m_config->m, m_quad);
    m_mesh->calcAreas(m_quad, m_config->m);
    qDebug() << "Here the zslice = " << m_mesh->zElemCt;
    geomDialog->updateMesh(m_mesh);

    //XSection *xs = new XSection(config);
    m_xs = new XSection(m_config);
    xsDialog->updateXs(m_xs);

    OutWriter::writeZoneId(std::string("zoneid.dat"), *m_mesh);

    outputDialog->updateMesh(m_mesh);
    //std::vector<float> solution = gssolver(m_quad, m_mesh, m_xs, m_config);
    std::vector<float> raysolution = raytrace(m_quad, m_mesh, m_xs, m_config);
    //std::vector<float> solution = gssolver(m_quad, m_mesh, m_xs, m_config, raysolution);
    outputDialog->updateSolution(raysolution);
}

MainWindow::~MainWindow()
{
    delete ui;

    if(m_config != NULL)
        delete m_config;

    if(m_mesh != NULL)
        delete m_mesh;

    if(m_xs != NULL)
        delete m_xs;

    if(m_quad != NULL)
        delete m_quad;
}

void MainWindow::launchSolver()
{
    std::vector<float> solution = gssolver(m_quad, m_mesh, m_xs, m_config, NULL);
    outputDialog->updateSolution(solution);
}

void MainWindow::userDebugNext()
{
    m_mutex.unlock();
}

void MainWindow::userDebugAbort()
{

}

QMutex &MainWindow::getBlockingMutex()
{
    return m_mutex;
}

void MainWindow::slotLoadConfigClicked()
{
    //QString filename = QFileDialog::getOpenFileName(this, "Open Config File", QDir::homePath(), "Config Files(*.cfg);;All Files (*)");
    QString filename = QFileDialog::getOpenFileName(this, "Open Config File", "/media/data/thesis/doctors/", "Config Files(*.cfg);;All Files (*)");

    if(!filename.isEmpty())
        m_config->loadFile(filename.toStdString());
}
