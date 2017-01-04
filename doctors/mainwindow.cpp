#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QDebug>
#include <QFileDialog>
#include <QDir>
#include <QThread>

#include "quadrature.h"
#include "mesh.h"
#include "xsection.h"

#include "gui/outputdialog.h"
#include "gui/geomdialog.h"
#include "gui/quaddialog.h"
#include "gui/xsectiondialog.h"
#include "outwriter.h"
#include "ctdatamanager.h"
#include "legendre.h"
#include "solver.h"

#include "outwriter.h"

#include "mcnpwriter.h"

#include "materialutils.h"  // TODO - Delete, not needed, just for testing

//#include "config.h"

QPalette *MainWindow::m_goodPalette = NULL;
QPalette *MainWindow::m_badPalette = NULL;

MainWindow::MainWindow(QWidget *parent):
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    //m_config(NULL),
    m_mesh(NULL),
    m_xs(NULL),
    m_quad(NULL),
    outputDialog(NULL),
    geomDialog(NULL),
    quadDialog(NULL),
    xsDialog(NULL),
    m_geomLoaded(false),
    m_xsLoaded(false),
    m_quadLoaded(false),
    m_paramsLoaded(false),
    m_configSelectDialog(NULL),
    m_pendingUserContinue(),
    m_mutex(),
    m_parser(NULL),
    m_xsWorkerThread(NULL),
    m_solver(NULL),
    m_solverWorkerThread(NULL),
    m_solution(NULL),
    m_raytrace(NULL)
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

    m_parser = new AmpxParser;

    m_solver = new Solver;

    // Connect explore buttons
    connect(ui->actionSolution_Explorer, SIGNAL(triggered()), outputDialog, SLOT(show()));
    connect(ui->quadExplorePushButton, SIGNAL(clicked()), quadDialog, SLOT(show()));
    connect(ui->geometryExplorePushButton, SIGNAL(clicked()), geomDialog, SLOT(show()));

    connect(m_parser, SIGNAL(signalNotifyNumberNuclides(int)), ui->mainProgressBar, SLOT(setMaximum(int)));

    connect(m_solver, SIGNAL(solverFinished(std::vector<float>*)), this, SLOT(onSolverFinished(std::vector<float>*)));
    connect(m_solver, SIGNAL(raytracerFinished(std::vector<float>*)), this, SLOT(onRaytracerFinished(std::vector<float>*)));

    // Set up xs reader threads
    m_parser->moveToThread(&m_xsWorkerThread);
    connect(&m_xsWorkerThread, SIGNAL(finished()), m_parser, SLOT(deleteLater()));
    connect(this, SIGNAL(signalBeginXsParse(QString)), m_parser, SLOT(parseFile(QString)));
    connect(m_parser, SIGNAL(signalXsUpdate(int)), this, SLOT(xsParseUpdateHandler(int)));
    connect(m_parser, SIGNAL(finishedParsing(AmpxParser*)), this, SLOT(buildMaterials(AmpxParser*)));
    m_xsWorkerThread.start();

    // Set up the solver thread
    m_solver->moveToThread(&m_solverWorkerThread);
    connect(&m_xsWorkerThread, SIGNAL(finished()), m_solver, SLOT(deleteLater()));
    connect(this, SIGNAL(signalLaunchRaytracer(const Quadrature*,const Mesh*,const XSection*)), m_solver, SLOT(raytraceIso(const Quadrature*,const Mesh*,const XSection*)));
    connect(this, SIGNAL(signalLaunchSolver(const Quadrature*,const Mesh*,const XSection*,const std::vector<float>*)), m_solver, SLOT(gssolverIso(const Quadrature*,const Mesh*,const XSection*,const std::vector<float>*)));
    connect(m_solver, SIGNAL(signalNewIteration(std::vector<float>*)), outputDialog, SLOT(reRender(std::vector<float>*)));
    m_solverWorkerThread.start();

    // Add the tooltips
    updateLaunchButton();  // Sets the tooltip

    ui->geometryOpenPushButton->setToolTip("Opens a dialog box to import a CT data file");

    ui->geometryGroupBox->setStyleSheet("QGroupBox { color: red; } ");
    ui->xsGroupBox->setStyleSheet("QGroupBox { color: red; } ");
    ui->paramsGroupBox->setStyleSheet("QGroupBox { color: red; } ");
    ui->quadGroupBox->setStyleSheet("QGroupBox { color: red; } ");


    // Make a configuration object and load its defaults
    //m_config = new Config;
    //m_config->loadDefaults();

    qDebug() << "Loaded default configuration";

    quadDialog->updateQuad(m_quad);
    m_xs = new XSection();

    AssocLegendre a;
    int l = 5;
    for(int m = 0; m <= l; m++)
        std::cout << a(l, m, 0.75) << std::endl;
}

MainWindow::~MainWindow()
{
    delete ui;

    //if(m_config != NULL)
    //    delete m_config;

    if(m_mesh != NULL)
        delete m_mesh;

    if(m_xs != NULL)
        delete m_xs;

    if(m_quad != NULL)
        delete m_quad;

    delete m_goodPalette;
    delete m_badPalette;

    m_xsWorkerThread.quit();
    m_xsWorkerThread.wait();
}

void MainWindow::on_launchSolverPushButton_clicked()
{
    outputDialog->show();

    // This can't be done without the energy group information
    m_mesh->calcAreas(m_quad, m_parser->getGammaEnergyGroups());

    // When the raytracer finishes, the gs solver is automatically launched
    emit signalLaunchRaytracer(m_quad, m_mesh, m_xs);
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

void MainWindow::on_geometryOpenPushButton_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this, "Open CT Data File", "/media/data/thesis/doctors/data/", "Binary Files(*.bin);;All Files (*)");

    if(filename.isEmpty())
    {
        qDebug() << "Error, failed to load CT Data file";
        m_geomLoaded = false;
        updateLaunchButton();
        return;
    }

    ui->geometryFileLineEdit->setText(filename);

    CtDataManager ctman;
    m_mesh = ctman.parse16(64, 64, 16, filename.toStdString());

    OutWriter::writeArray("/media/Storage/thesis/mcnp.gitignore/ctdensity.dat", m_mesh->density);

    if(m_mesh == NULL)
    {
        qDebug() << "Error, failed to parse CT Data file";
        m_geomLoaded = false;
        updateLaunchButton();
        return;
    }

    qDebug() << "Here the zslice = " << m_mesh->zElemCt;
    geomDialog->updateMesh(m_mesh);
    outputDialog->updateMesh(m_mesh);
    m_geomLoaded = true;

    updateLaunchButton();
}

void MainWindow::updateLaunchButton()
{
    // TODO - temporarily removed the dependence on the solver params since I'm hard coding them for now
    //if(m_geomLoaded && m_quadLoaded && m_xsLoaded && m_paramsLoaded)
    if(m_geomLoaded && m_quadLoaded && m_xsLoaded)
    {
        ui->launchSolverPushButton->setEnabled(true);

        McnpWriter mcnpwriter;
        mcnpwriter.writeMcnp("../mcnp.gitignore/mcnp_out.inp", m_mesh, false);
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
        ui->quadGroupBox->setStyleSheet("QGroupBox { color: black; } ");
        ui->quadExplorePushButton->setEnabled(true);
    }
    else
    {
        ui->quadGroupBox->setStyleSheet("QGroupBox { color: red; } ");
        ui->quadExplorePushButton->setEnabled(false);
    }

    if(m_xsLoaded)
    {
        ui->xsGroupBox->setStyleSheet("QGroupBox { color: black; } ");
        ui->xsExplorePushButton->setEnabled(true);
    }
    else
    {
        ui->xsGroupBox->setStyleSheet("QGroupBox { color: red; } ");
        ui->xsExplorePushButton->setEnabled(false);
    }

    if(m_paramsLoaded)
    {
        ui->paramsGroupBox->setStyleSheet("QGroupBox { color: black; } ");
        ui->paramsExplorePushButton->setEnabled(true);
    }
    else
    {
        ui->paramsGroupBox->setStyleSheet("QGroupBox { color: red; } ");
        ui->paramsExplorePushButton->setEnabled(false);
    }

    ui->geometryGroupBox->update();

}

void MainWindow::on_quadTypeComboBox_activated(int type)
{
    switch(type)
    {
    case 0:  // Sn
        ui->quadData1ComboBox->setEnabled(true);
        ui->quadData1ComboBox->clear();
        ui->quadData1ComboBox->addItem("N");
        ui->quadData1ComboBox->addItem("2");
        ui->quadData1ComboBox->addItem("4");
        ui->quadData1ComboBox->addItem("6");
        ui->quadData2ComboBox->clear();
        ui->quadData2ComboBox->setEnabled(false);
        ui->quadOpenPushButton->setEnabled(false);
        ui->quadFileLineEdit->setEnabled(false);
        break;
    case 1:  // Custom
        ui->quadOpenPushButton->setEnabled(true);
        ui->quadFileLineEdit->setEnabled(true);
        ui->quadData1ComboBox->clear();
        ui->quadData1ComboBox->setEnabled(false);
        ui->quadData2ComboBox->clear();
        ui->quadData2ComboBox->setEnabled(false);
        m_quadLoaded = false;
        updateLaunchButton();
        break;
    case 2:
        ui->quadOpenPushButton->setEnabled(false);
        ui->quadFileLineEdit->setEnabled(false);
        ui->quadData1ComboBox->clear();
        ui->quadData1ComboBox->setEnabled(true);
        ui->quadData1ComboBox->addItem("Select...");
        ui->quadData1ComboBox->addItem("Unidirectional");
        ui->quadData2ComboBox->clear();
        ui->quadData2ComboBox->setEnabled(false);
        m_quadLoaded = false;
        updateLaunchButton();
        break;
    default:
        qDebug() << "Illegal combo box (type = " << type << ")selection";
    }
}

void MainWindow::on_quadData1ComboBox_activated(int d1)
{
    switch(ui->quadTypeComboBox->currentIndex())
    {
    case 0: // Sn
        if(d1 == 0)
        {
            m_quadLoaded = false;
            updateLaunchButton();
        }
        else
        {
            m_quadLoaded = true;
            m_quad = new Quadrature(ui->quadData1ComboBox->itemText(d1).toInt());
            updateLaunchButton();
        }
        break;
    case 1: // Custom
        break;
    case 2: // Debug
        if(d1 == 0)  // Unidirectional
        {
            m_quadLoaded = false;
            updateLaunchButton();
        }
        else
        {
            if(m_quad != NULL)
                delete m_quad;
            m_quad = new Quadrature;
            m_quad->loadSpecial(3);
            m_quadLoaded = true;
            updateLaunchButton();
        }
        break;
    default:
        qDebug() << "Illegal combo box combination, Quad type cannot have data1 values";
    }
}

void MainWindow::on_quadData2ComboBox_activated(int)
{

}


void MainWindow::on_xsOpenPushButton_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this, "Open AMPX Formatted Data File", "/media/data/thesis/doctors/data/", "AMPX (*.ampx);;All Files (*)");

    if(filename.isEmpty())
    {
        qDebug() << "Error, failed to load cross section Data file";
        m_xsLoaded = false;
        updateLaunchButton();
        return;
    }

    ui->xsFileLineEdit->setText(filename);

    m_xsLoaded = true;

    updateLaunchButton();

    emit signalBeginXsParse(filename);
}

void MainWindow::on_xsExplorePushButton_clicked()
{
    xsDialog->setXs(m_parser);
    xsDialog->show();
}

void MainWindow::xsParseErrorHandler(QString msg)
{
    qDebug() << "Error: " << msg;
}

void MainWindow::xsParseUpdateHandler(int x)
{
    ui->mainProgressBar->setValue(x+1);
    // TODO: should catch signal emitted from the AmpxParser
}

bool MainWindow::buildMaterials(AmpxParser *parser)
{
    qDebug() << "Generating materials";

    // One extra material for the empty material at the end
    m_xs->allocateMemory(static_cast<unsigned int>(MaterialUtils::hounsfieldRangePhantom19Elements.size()+1), parser->getGammaEnergyGroups(), 6);

    bool allPassed = true;

    // Add the materials to the xs library
    for(unsigned int i = 0; i < MaterialUtils::hounsfieldRangePhantom19Elements.size(); i++)
        allPassed &= m_xs->addMaterial(MaterialUtils::hounsfieldRangePhantom19Elements, MaterialUtils::hounsfieldRangePhantom19Weights[i], parser);

    // The last material is empty and should never be used
    allPassed &= m_xs->addMaterial(std::vector<int>{}, std::vector<float>{}, parser);

    return allPassed;
}

void MainWindow::onRaytracerFinished(std::vector<float>* uncollided)
{
    m_raytrace = uncollided;

    OutWriter::writeArray("uncol_flux.dat", *uncollided);

    emit signalLaunchSolver(m_quad, m_mesh, m_xs, uncollided);
}

void MainWindow::onSolverFinished(std::vector<float> *solution)
{
    m_solution = solution;

    OutWriter::writeArray("solution.dat", *solution);

    std::vector<float> x;
    std::vector<float> matids;
    std::vector<float> density;
    std::vector<float> flux;

    std::vector<std::vector<float> > push;

    for(unsigned int i = 0; i < m_mesh->xElemCt; i++)
    {
        float centerpt = (m_mesh->xNodes[i] + m_mesh->xNodes[i+1])/2.0;
        x.push_back(centerpt);
        matids.push_back(m_mesh->getZoneIdAt(i, 5, 9));
        density.push_back(m_mesh->getPhysicalDensityAt(i, 5, 9));
        flux.push_back((*m_solution)[18*m_mesh->voxelCount() + m_mesh->getFlatIndex(i, 5, 9)]);
    }

    push.push_back(x);
    push.push_back(matids);
    push.push_back(density);
    push.push_back(flux);

    OutWriter::writeFloatArrays("/media/Storage/thesis/doctors/solution.dat", push);
}
