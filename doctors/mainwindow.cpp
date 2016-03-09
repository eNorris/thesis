#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QDebug>

#include "config.h"
#include "quadrature.h"
#include "mesh.h"
#include "xsection.h"
#include "solvers.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // Make a configuration object and load its defaults
    Config config;
    config.loadDefaults();

    qDebug() << "Loaded default configuration";

    Quadrature quad(config);

    Mesh mesh(config, quad);

    XSection xs(config);

    std::vector<float> solution = gssolver(quad, mesh, xs);
}

MainWindow::~MainWindow()
{
    delete ui;
}
