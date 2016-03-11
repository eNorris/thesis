#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QDebug>

#include "outputdialog.h"
#include "geomdialog.h"
#include "quaddialog.h"
#include "xsectiondialog.h"

#include "config.h"
#include "solvers.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
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

    connect(this, SIGNAL(signalNewIteration(std::vector<float>)), outputDialog, SLOT(disp(std::vector<float>)));

    // Make a configuration object and load its defaults
    Config config;
    config.loadDefaults();

    qDebug() << "Loaded default configuration";

    Quadrature quad(config);
    quadDialog->updateQuad(&quad);

    Mesh mesh(config, quad);
    geomDialog->updateMesh(&mesh);

    XSection xs(config);
    xsDialog->updateXs(&xs);

    std::vector<float> solution = gssolver(quad, mesh, xs);
}

MainWindow::~MainWindow()
{
    delete ui;
}
