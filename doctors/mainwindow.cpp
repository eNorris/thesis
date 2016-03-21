#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QDebug>

#include "quadrature.h"
#include "mesh.h"
#include "xsection.h"

#include "outputdialog.h"
#include "geomdialog.h"
#include "quaddialog.h"
#include "xsectiondialog.h"

#include "config.h"
#include "solvers.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    m_config(NULL),
    m_mesh(NULL),
    m_xs(NULL),
    m_quad(NULL)
{
    ui->setupUi(this);

    outputDialog = new OutputDialog();
    geomDialog = new GeomDialog();
    quadDialog = new QuadDialog();
    xsDialog = new XSectionDialog();

    connect(ui->actionSolution_Explorer, SIGNAL(triggered()), outputDialog, SLOT(show()));

    connect(ui->selectQuadPushButton, SIGNAL(clicked()), quadDialog, SLOT(show()));

    connect(ui->buildGeomPushButton, SIGNAL(clicked()), geomDialog, SLOT(show()));

    connect(ui->loadXsPushButton, SIGNAL(clicked()), xsDialog, SLOT(show()));

    connect(ui->launchSolverPushButton, SIGNAL(clicked()), this, SLOT(launchSolver()));

    //connect(this, SIGNAL(signalNewIteration(std::vector<float>)), outputDialog, SLOT(disp(std::vector<float>)));
    // TODO - Update like above at some point
    connect(this, SIGNAL(signalNewIteration(std::vector<float>)), outputDialog, SLOT(reRender(std::vector<float>)));

    // Make a configuration object and load its defaults
    //config = new Config;
    m_config = new Config;
    m_config->loadDefaults();

    qDebug() << "Loaded default configuration";

    //Quadrature *m_quad = new Quadrature(config);
    m_quad = new Quadrature(m_config);
    quadDialog->updateQuad(m_quad);

    //Mesh *mesh = new Mesh(config, quad);
    m_mesh = new Mesh(m_config, m_quad);
    qDebug() << "Here the zslice = " << m_mesh->zMesh;
    geomDialog->updateMesh(m_mesh);

    //XSection *xs = new XSection(config);
    m_xs = new XSection(m_config);
    xsDialog->updateXs(m_xs);

    outputDialog->updateMesh(m_mesh);
    std::vector<float> solution = gssolver(m_quad, m_mesh, m_xs, m_config);
    outputDialog->updateSolution(solution);
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
    //qDebug() << "Solver isn't implemented yet!";

    std::vector<float> solution = gssolver(m_quad, m_mesh, m_xs, m_config);
    outputDialog->updateSolution(solution);
}
