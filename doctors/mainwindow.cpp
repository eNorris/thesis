#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // Make a configuration object and load its defaults
    Config config;
    config.loadDefaults();
}

MainWindow::~MainWindow()
{
    delete ui;
}
