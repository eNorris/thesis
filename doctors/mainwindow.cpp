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
    config(NULL)
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
    config = new Config;
    config->loadDefaults();

    qDebug() << "Loaded default configuration";

    Quadrature *quad = new Quadrature(config);
    quadDialog->updateQuad(quad);

    Mesh *mesh = new Mesh(config, quad);
    qDebug() << "Here the zslice = " << mesh->zMesh;
    geomDialog->updateMesh(mesh);

    XSection *xs = new XSection(config);
    xsDialog->updateXs(xs);

    outputDialog->updateMesh(mesh);
    std::vector<float> solution = gssolver(quad, mesh, xs, config);
    outputDialog->updateSolution(solution);
}

MainWindow::~MainWindow()
{
    delete ui;

    if(config != NULL)
        delete config;
}

void MainWindow::launchSolver()
{
    qDebug() << "Solver isn't implemented yet!";
}
