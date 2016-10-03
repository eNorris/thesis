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

QPalette *MainWindow::m_goodPalette = NULL;
QPalette *MainWindow::m_badPalette = NULL;

MainWindow::MainWindow(QWidget *parent):
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    m_config(NULL),
    m_mesh(NULL),
    m_xs(NULL),
    m_quad(NULL),
    m_configSelectDialog(NULL),
    m_geomLoaded(false),
    m_xsLoaded(false),
    m_quadLoaded(false),
    m_paramsLoaded(false)
{
    ui->setupUi(this);

    // Initialize static member variables
    MainWindow::m_badPalette = new QPalette();
    m_badPalette->setColor(QPalette::Text, Qt::red);
    MainWindow::m_goodPalette = new QPalette();
    m_goodPalette->setColor(QPalette::Text, Qt::green);

    outputDialog = new OutputDialog(this);
    geomDialog = new GeomDialog(this);
    quadDialog = new QuadDialog(this);
    xsDialog = new XSectionDialog(this);

    m_configSelectDialog = new QFileDialog(this);
    m_configSelectDialog->setAcceptMode(QFileDialog::AcceptOpen);
    m_configSelectDialog->setFileMode(QFileDialog::ExistingFile);

    connect(ui->actionSolution_Explorer, SIGNAL(triggered()), outputDialog, SLOT(show()));

    //connect(ui->selectQuadPushButton, SIGNAL(clicked()), quadDialog, SLOT(show()));
    connect(ui->quadExplorePushButton, SIGNAL(clicked()), quadDialog, SLOT(show()));

    //connect(ui->buildGeomPushButton, SIGNAL(clicked()), geomDialog, SLOT(show()));
    connect(ui->geometryExplorePushButton, SIGNAL(clicked()), geomDialog, SLOT(show()));
    connect(ui->geometryOpenPushButton, SIGNAL(clicked()), this, SLOT(slotOpenCtData()));

    //connect(ui->loadXsPushButton, SIGNAL(clicked()), xsDialog, SLOT(show()));

    connect(ui->launchSolverPushButton, SIGNAL(clicked()), this, SLOT(launchSolver()));

    //connect(ui->loadInputPushButton, SIGNAL(clicked()), this, SLOT(slotLoadConfigClicked()));

    //connect(this, SIGNAL(signalNewIteration(std::vector<float>)), outputDialog, SLOT(disp(std::vector<float>)));
    // TODO - Update like above at some point
    connect(this, SIGNAL(signalNewIteration(std::vector<float>)), outputDialog, SLOT(reRender(std::vector<float>)));

    connect(ui->quadTypecomboBox, SIGNAL(activated(int)), this, SLOT(slotQuadSelected(int)));
    connect(ui->quadData1ComboBox, SIGNAL(activated(int)), this, SLOT(slotQuadSelected(int)));
    connect(ui->quadData2ComboBox, SIGNAL(activated(int)), this, SLOT(slotQuadSelected(int)));

    // Add the tooltips
    //ui->launchSolverPushButton->setToolTip("");
    updateLaunchButton();  // Sets the tooltip

    ui->geometryOpenPushButton->setToolTip("Opens a dialog box to import a CT data file");

    ui->geometryGroupBox->setStyleSheet("QGroupBox { color: red; } ");
    ui->xsGroupBox->setStyleSheet("QGroupBox { color: red; } ");
    ui->paramsGroupBox->setStyleSheet("QGroupBox { color: red; } ");
    ui->quadGroupBox->setStyleSheet("QGroupBox { color: red; } ");


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


    //XSection *xs = new XSection(config);
    m_xs = new XSection(m_config);
    xsDialog->updateXs(m_xs);

    //OutWriter::writeZoneId(std::string("zoneid.dat"), *m_mesh);

    //outputDialog->updateMesh(m_mesh);
    //std::vector<float> solution = gssolver(m_quad, m_mesh, m_xs, m_config);
    //std::vector<float> raysolution = raytrace(m_quad, m_mesh, m_xs, m_config);
    //std::vector<float> solution = gssolver(m_quad, m_mesh, m_xs, m_config, raysolution);
    //outputDialog->updateSolution(raysolution);
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

    delete m_goodPalette;
    delete m_badPalette;
}

void MainWindow::launchSolver()
{
    m_mesh->calcAreas(m_quad, m_xs->groupCount());
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

/*
void MainWindow::slotLoadConfigClicked()
{
    //QString filename = QFileDialog::getOpenFileName(this, "Open Config File", QDir::homePath(), "Config Files(*.cfg);;All Files (*)");
    QString filename = QFileDialog::getOpenFileName(this, "Open Config File", "/media/data/thesis/doctors/", "Config Files(*.cfg);;All Files (*)");

    if(!filename.isEmpty())
        m_config->loadFile(filename.toStdString());
}
*/

void MainWindow::slotOpenCtData()
{
    QString filename = QFileDialog::getOpenFileName(this, "Open CT Data File", "/media/data/thesis/doctors/", "Binary Files(*.bin);;All Files (*)");

    //if(!filename.isEmpty())
    //    m_config->loadFile(filename.toStdString());

    if(filename.isEmpty())
    {
        qDebug() << "Error, failed to load CT Data file";
        m_geomLoaded = false;
        updateLaunchButton();
        return;
    }

    ui->geometryFileLineEdit->setText(filename);

    CtDataManager ctman;
    m_mesh = ctman.parse16(256, 256, 64, filename.toStdString());

    if(m_mesh == NULL)
    {
        qDebug() << "Error, failed to parse CT Data file";
        m_geomLoaded = false;
        updateLaunchButton();
        return;
    }

    //m_mesh->calcAreas(m_quad, m_config->m);
    qDebug() << "Here the zslice = " << m_mesh->zElemCt;
    geomDialog->updateMesh(m_mesh);
    outputDialog->updateMesh(m_mesh);
    m_geomLoaded = true;

    updateLaunchButton();
}

void MainWindow::updateLaunchButton()
{
    if(m_geomLoaded && m_quadLoaded && m_xsLoaded && m_paramsLoaded)
    {
        ui->launchSolverPushButton->setEnabled(true);
        //ui->launchSolverPushButton->setToolTip("Ready to launch solver!");
    }
    else
    {
        ui->launchSolverPushButton->setEnabled(false);
    }

    if(m_geomLoaded)
    {
        ui->geometryGroupBox->setStyleSheet("QGroupBox { color: black; } ");
        ui->geometryExplorePushButton->setEnabled(true);
    }
    else
    {
        ui->geometryGroupBox->setStyleSheet("QGroupBox { color: red; } ");
        ui->geometryExplorePushButton->setEnabled(false);
    }

    if(m_quadLoaded)
    {
        //ui->geometryGroupBox->setPalette(*m_goodPalette);
        ui->quadGroupBox->setStyleSheet("QGroupBox { color: black; } ");
        ui->quadExplorePushButton->setEnabled(true);
    }
    else
    {
        //ui->geometryGroupBox->setPalette(*m_badPalette);
        ui->quadGroupBox->setStyleSheet("QGroupBox { color: red; } ");
        ui->quadExplorePushButton->setEnabled(false);
    }

    if(m_xsLoaded)
    {
        //ui->geometryGroupBox->setPalette(*m_goodPalette);
        ui->xsGroupBox->setStyleSheet("QGroupBox { color: black; } ");
        ui->xsExplorePushButton->setEnabled(true);
    }
    else
    {
        //ui->geometryGroupBox->setPalette(*m_badPalette);
        ui->xsGroupBox->setStyleSheet("QGroupBox { color: red; } ");
        ui->xsExplorePushButton->setEnabled(false);
    }

    if(m_paramsLoaded)
    {
        //ui->geometryGroupBox->setPalette(*m_goodPalette);
        ui->paramsGroupBox->setStyleSheet("QGroupBox { color: black; } ");
        ui->paramsExplorePushButton->setEnabled(true);
    }
    else
    {
        //ui->geometryGroupBox->setPalette(*m_badPalette);
        ui->paramsGroupBox->setStyleSheet("QGroupBox { color: red; } ");
        ui->paramsExplorePushButton->setEnabled(false);
    }

    ui->geometryGroupBox->update();

}

void MainWindow::slotQuadSelected(int)
{
    int type = ui->quadTypecomboBox->currentIndex();
    int d1 = ui->quadData1ComboBox->currentIndex();
    int d2 = ui->quadData2ComboBox->currentIndex();

    if(type == 0)  // Sn
    {
        ui->quadData1ComboBox->setEnabled(true);
        ui->quadData1ComboBox->clear();
        ui->quadData1ComboBox->addItem("N");
        ui->quadData1ComboBox->addItem("2");
        ui->quadData1ComboBox->addItem("4");
        ui->quadData1ComboBox->addItem("8");
        ui->quadData2ComboBox->clear();
        ui->quadData2ComboBox->setEnabled(false);
        ui->quadOpenPushButton->setEnabled(false);
        ui->quadFileLineEdit->setEnabled(false);
    }
    else if(type == 1)  // Custom
    {
        ui->quadOpenPushButton->setEnabled(true);
        ui->quadFileLineEdit->setEnabled(true);
    }
    else
    {
        qDebug() << "Illegal combo box (type = " << type << ")selection";
    }

    qDebug() << "type = " << type << "  d1 = " << d1 << "   d2 = " << d2;
}
