#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QDebug>
#include <QFileDialog>
#include <QDir>
#include <QThread>

#include "quadrature.h"
#include "mesh.h"
#include "xsection.h"
#include "solverparams.h"
#include "sourceparams.h"

#include "gui/outputdialog.h"
#include "gui/geomdialog.h"
#include "gui/quaddialog.h"
#include "gui/xsectiondialog.h"
#include "gui/energydialog.h"

#include "outwriter.h"
#include "ctdatamanager.h"
#include "legendre.h"
#include "solver.h"
#include "mcnpwriter.h"
#include "materialutils.h"

#undef SLOT
#define _SLOT(a) "1"#a
#define SLOT(a) _SLOT(a)

#undef SIGNAL
#define _SIGNAL(a) "2"#a
#define SIGNAL(a) _SIGNAL(a)

QPalette *MainWindow::m_goodPalette = NULL;
QPalette *MainWindow::m_badPalette = NULL;

MainWindow::MainWindow(QWidget *parent):
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    m_mesh(NULL),
    m_xs(NULL),
    m_quad(NULL),
    m_srcParams(NULL),
    m_solvParams(NULL),
    outputDialog(NULL),
    geomDialog(NULL),
    quadDialog(NULL),
    xsDialog(NULL),
    energyDialog(NULL),
    m_geomLoaded(false),
    m_xsLoaded(false),
    m_quadLoaded(false),
    m_solverLoaded(false),
    m_srcLoaded(false),
    m_configSelectDialog(NULL),
    m_parser(NULL),
    m_xsWorkerThread(NULL),
    m_ctman(NULL),
    m_meshWorkerThread(NULL),
    m_solver(NULL),
    m_solverWorkerThread(NULL),
    m_solution(NULL),
    m_raytrace(NULL),
    m_solType(MainWindow::ISOTROPIC),
    m_pn(0)
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
    energyDialog = new EnergyDialog(this);

    m_configSelectDialog = new QFileDialog(this);
    m_configSelectDialog->setAcceptMode(QFileDialog::AcceptOpen);
    m_configSelectDialog->setFileMode(QFileDialog::ExistingFile);

    m_parser = new AmpxParser;
    m_solver = new Solver;
    m_xs = new XSection;
    m_solvParams = new SolverParams;
    m_ctman = new CtDataManager;

    // Connect explore buttons
    connect(ui->actionSolution_Explorer, SIGNAL(triggered()), outputDialog, SLOT(show()));
    connect(ui->quadExplorePushButton, SIGNAL(clicked()), quadDialog, SLOT(show()));
    connect(ui->geometryExplorePushButton, SIGNAL(clicked()), geomDialog, SLOT(show()));
    connect(ui->sourceEnergyPushButton, SIGNAL(clicked()), energyDialog, SLOT(show()));

    // Set up xs reader threads
    m_parser->moveToThread(&m_xsWorkerThread);
    connect(&m_xsWorkerThread, SIGNAL(finished()), m_parser, SLOT(deleteLater()));
    connect(this, SIGNAL(signalBeginXsParse(QString)), m_parser, SLOT(parseFile(QString)));
    connect(m_parser, SIGNAL(signalXsUpdate(int)), this, SLOT(xsParseUpdateHandler(int)));
    connect(m_parser, SIGNAL(signalNotifyNumberNuclides(int)), ui->mainProgressBar, SLOT(setMaximum(int)));
    connect(m_parser, SIGNAL(finishedParsing(AmpxParser*)), this, SLOT(xsParseFinished(AmpxParser*)));
    m_xsWorkerThread.start();

    // Set up mesh reader threads
    m_ctman->moveToThread(&m_meshWorkerThread);
    connect(&m_meshWorkerThread, SIGNAL(finished()), m_ctman, SLOT(deleteLater()));
    connect(this, SIGNAL(signalBeginMeshParse(int,int,int,QString)), m_ctman, SLOT(parse16(int,int,int,QString)));
    connect(m_ctman, SIGNAL(signalMeshUpdate(int)), this, SLOT(meshParseUpdateHandler(int)));
    //connect(m_ctman, SIGNAL(signal(int)), ui->geomProgressBar, SLOT(setMaximum(int)));
    connect(m_ctman, SIGNAL(finishedMeshParsing(Mesh*)), this, SLOT(meshParseFinished(Mesh*)));
    m_meshWorkerThread.start();

    // Set up the solver thread
    m_solver->moveToThread(&m_solverWorkerThread);
    connect(&m_xsWorkerThread, SIGNAL(finished()), m_solver, SLOT(deleteLater()));

    connect(this, SIGNAL(signalLaunchRaytracerIso(const Quadrature*,const Mesh*,const XSection*,const SolverParams*,const SourceParams*)), m_solver, SLOT(raytraceIso(const Quadrature*,const Mesh*,const XSection*,const SolverParams*,const SourceParams*)));
    connect(this, SIGNAL(signalLaunchSolverIso(const Quadrature*,const Mesh*,const XSection*,const SolverParams*, const SourceParams*,const std::vector<RAY_T>*)), m_solver, SLOT(gsSolverIso(const Quadrature*,const Mesh*,const XSection*,const SolverParams*, const SourceParams*,const std::vector<RAY_T>*)));

    connect(this, SIGNAL(signalLaunchRaytracerLegendre(const Quadrature*,const Mesh*,const XSection*, const SolverParams*,const SourceParams*)), m_solver, SLOT(raytraceLegendre(const Quadrature*,const Mesh*,const XSection*, const SolverParams*,const SourceParams*)));
    connect(this, SIGNAL(signalLaunchSolverLegendre(const Quadrature*,const Mesh*,const XSection*,const SolverParams*, const SourceParams*,const std::vector<RAY_T>*)), m_solver, SLOT(gsSolverLegendre(const Quadrature*,const Mesh*,const XSection*,const SolverParams*,const SourceParams*,const std::vector<RAY_T>*)));

    connect(this, SIGNAL(signalLaunchRaytracerHarmonic(const Quadrature*,const Mesh*,const XSection*, const SolverParams*,const SourceParams*)), m_solver, SLOT(raytraceHarmonic(const Quadrature*,const Mesh*,const XSection*,const SolverParams*,const SourceParams*)));
    connect(this, SIGNAL(signalLaunchSolverHarmonic(const Quadrature*,const Mesh*,const XSection*,const SolverParams*, const SourceParams*,const std::vector<RAY_T>*)), m_solver, SLOT(gsSolverHarmonic(const Quadrature*,const Mesh*,const XSection*,const SolverParams*,const SourceParams*,const std::vector<RAY_T>*)));

    connect(energyDialog, SIGNAL(notifyOkClicked()), this, SLOT(energyGroupsUpdated()));

    connect(m_solver, SIGNAL(signalRaytracerFinished(std::vector<RAY_T>*)), this, SLOT(onRaytracerFinished(std::vector<RAY_T>*)));
    connect(m_solver, SIGNAL(signalSolverFinished(std::vector<SOL_T>*)), this, SLOT(onSolverFinished(std::vector<SOL_T>*)));

    connect(m_solver, SIGNAL(signalNewRaytracerIteration(std::vector<RAY_T>*)), outputDialog, SLOT(updateRaytracerData(std::vector<RAY_T>*)));
    connect(m_solver, SIGNAL(signalNewSolverIteration(std::vector<SOL_T>*)), outputDialog, SLOT(updateSolverData(std::vector<SOL_T>*)));
    m_solverWorkerThread.start();

    for(int i = 0; i < 10; i++)
    {
        ui->paramsPnComboBox->addItem(QString::number(i));
    }

    // Add the tooltips
    updateLaunchButton();  // Sets the tooltip

    ui->geometryOpenPushButton->setToolTip("Opens a dialog box to import a CT data file");

    ui->sourceImageLabel->setPixmapFile("/media/Storage/thesis/doctors/images/source1.png");
    //QPixmap srcImg("/media/Storage/thesis/doctors/images/source1.png");
    //ui->sourceImageLabel->setScaledContents(true);
    //ui->sourceImageLabel->setPixmap(srcImg.scaledToWidth(ui->sourceImageLabel->width(), Qt::SmoothTransformation));

    //ui->geometryGroupBox->setStyleSheet("QGroupBox { color: red; } ");
    //ui->xsGroupBox->setStyleSheet("QGroupBox { color: red; } ");
    //ui->paramsGroupBox->setStyleSheet("QGroupBox { color: red; } ");
    //ui->quadGroupBox->setStyleSheet("QGroupBox { color: red; } ");

    ui->tabWidget->tabBar()->setTabTextColor(0, Qt::red);
    ui->tabWidget->tabBar()->setTabTextColor(1, Qt::red);
    ui->tabWidget->tabBar()->setTabTextColor(2, Qt::red);
    ui->tabWidget->tabBar()->setTabTextColor(3, Qt::red);
    ui->tabWidget->tabBar()->setTabTextColor(4, Qt::red);

    // Add unicode Greek text
    const QChar phySymbol(0x03C6);
    const QChar thetaSymbol(0x03B8);
    ui->sourcePhiLabel->setText(phySymbol);
    ui->sourceThetaLabel->setText(thetaSymbol);
    ui->sourcePhiLabel->setFont(QFont("Times New Roman"));
    ui->sourceThetaLabel->setFont(QFont("Times New Roman"));

    quadDialog->updateQuad(m_quad);
}

MainWindow::~MainWindow()
{
    delete ui;

    delete outputDialog;
    delete geomDialog;
    delete quadDialog;
    delete xsDialog;
    delete energyDialog;

    if(m_mesh != NULL)
        delete m_mesh;

    if(m_xs != NULL)
        delete m_xs;

    if(m_quad != NULL)
        delete m_quad;

    if(m_parser != NULL)
        delete m_parser;

    if(m_solvParams != NULL)
        delete m_solvParams;


    if(m_srcParams != NULL)
        delete m_srcParams;

    delete m_goodPalette;
    delete m_badPalette;

    // Kill running threads
    m_xsWorkerThread.quit();
    m_xsWorkerThread.wait();

    m_solverWorkerThread.quit();
    m_solverWorkerThread.wait();
}

void MainWindow::on_launchSolverPushButton_clicked()
{
    outputDialog->show();

    // This can't be done without the energy group information
    m_mesh->calcAreas(m_quad, m_parser->getGammaEnergyGroups());

    // Needs XS and Pn information
    if(!buildMaterials(m_parser))
    {
        QString errmsg = QString("One or more materials failed to be built properly. ");
        errmsg += "You may either abort the run or ignore this warning (which may produce incorrect results).";
        int resp = QMessageBox::warning(this, "Internal Error", errmsg, QMessageBox::Abort | QMessageBox::Ignore);
        if(resp == QMessageBox::Abort)
            return;
    }

    unsigned int srcType = ui->sourceTypeComboBox->currentIndex();
    if((srcType == 2 || srcType == 4) && ui->sourceNSpinBox->value() < 2)
    {
        QString errmsg = QString("You don't have at least 2 fan beams on a multi-fan source. ");
        errmsg += "You may either abort the run or ignore this warning (which may produce incorrect results).";
        int resp = QMessageBox::warning(this, "Internal Error", errmsg, QMessageBox::Abort | QMessageBox::Ignore);
        if(resp == QMessageBox::Abort)
            return;
    }

    if(m_srcParams == NULL)
    {
        QString errmsg = QString("The energy spectra must be loaded before the solver can be run");
        QMessageBox::warning(this, "Insufficient Data", errmsg, QMessageBox::Close);
        return;
    }

    if(!m_srcParams->update(energyDialog->getUserIntensity(), ui->sourceXDoubleSpinBox->value(), ui->sourceYDoubleSpinBox->value(), ui->sourceZDoubleSpinBox->value(),
                            ui->sourcePhiSpinBox->value(), ui->sourceThetaSpinBox->value(), ui->sourceNSpinBox->value(), ui->sourceDegreesCheckBox->isChecked(),
                            ui->sourceDistanceSpinBox->value(), ui->sourceWidthSpinBox->value(), ui->sourceHeightSpinBox->value(),
                            ui->sourceTypeComboBox->currentIndex()))
    {
        QString errmsg = QString("Could not update the energy intensity data. The energy structure may have changed.");
        QMessageBox::warning(this, "Invalid Vector Size", errmsg, QMessageBox::Close);
        return;
    }

    if(!m_srcParams->normalize())
    {
        QString errmsg = QString("The energy spectrum total is zero.");
        QMessageBox::warning(this, "Divide by zero", errmsg, QMessageBox::Close);
        return;
    }

    // When the raytracer finishes, the gs solver is automatically launched
    switch(m_solType)
    {
    case MainWindow::ISOTROPIC:
        emit signalLaunchRaytracerIso(m_quad, m_mesh, m_xs, m_solvParams, m_srcParams);
        break;
    case MainWindow::LEGENDRE:
        emit signalLaunchRaytracerLegendre(m_quad, m_mesh, m_xs, m_solvParams, m_srcParams);
        break;
    case MainWindow::HARMONIC:
        emit signalLaunchRaytracerHarmonic(m_quad, m_mesh, m_xs, m_solvParams, m_srcParams);
        break;
    default:
        qDebug() << "No solver of type " << m_solType;
    }
    return;
}

void MainWindow::on_geometryOpenPushButton_clicked()
{
    if(m_mesh != NULL)
    {
        delete m_mesh;
        m_mesh = NULL;
    }

    QString filename = QFileDialog::getOpenFileName(this, "Open CT Data File", "/media/data/thesis/doctors/data/", "Binary Files(*.bin);;All Files (*)");

    if(filename.isEmpty())
    {
        qDebug() << "Error, failed to load CT Data file";
        m_geomLoaded = false;
        updateLaunchButton();
        return;
    }

    ui->geometryFileLineEdit->setText(filename);

    //CtDataManager ctman;
    if(ui->signedCheckBox->isChecked())
    {
        if(ui->bitSpinBox->value() == 16)
        {
            ui->geomProgressBar->setMaximum(ui->xBinSpinBox->value() * ui->yBinSpinBox->value() * ui->zBinSpinBox->value());
            //m_mesh = m_ctman->parse16(ui->xBinSpinBox->value(), ui->yBinSpinBox->value(), ui->zBinSpinBox->value(), filename.toStdString());
            emit signalBeginMeshParse(ui->xBinSpinBox->value(), ui->yBinSpinBox->value(), ui->zBinSpinBox->value(), filename);
        }
        else
        {
            QString errmsg = "Only 16 bit (unsigned) binaries are currently supported";
            QMessageBox::warning(NULL, "Not Supported", errmsg, QMessageBox::Close);
        }
    }
    else
    {
        QString errmsg = "Only (16 bit) unsigned binaries are currently supported";
        QMessageBox::warning(NULL, "Not Supported", errmsg, QMessageBox::Close);
    }
}

void MainWindow::updateLaunchButton()
{
    if(m_geomLoaded && m_quadLoaded && m_xsLoaded && m_solverLoaded && m_srcLoaded)
    {
        ui->launchSolverPushButton->setEnabled(true);
    }
    else
    {
        ui->launchSolverPushButton->setEnabled(false);
    }

    if(m_geomLoaded)
    {
        //ui->geometryGroupBox->setStyleSheet("QGroupBox { color: black; } ");
        ui->tabWidget->tabBar()->setTabTextColor(0, Qt::black);
        ui->geometryExplorePushButton->setEnabled(true);
    }
    else
    {
        //ui->geometryGroupBox->setStyleSheet("QGroupBox { color: red; } ");
        ui->tabWidget->tabBar()->setTabTextColor(0, Qt::red);
        ui->geometryExplorePushButton->setEnabled(false);
    }

    if(m_xsLoaded)
    {
        //ui->xsGroupBox->setStyleSheet("QGroupBox { color: black; } ");
        ui->tabWidget->tabBar()->setTabTextColor(1, Qt::black);
        ui->xsExplorePushButton->setEnabled(true);
    }
    else
    {
        //ui->xsGroupBox->setStyleSheet("QGroupBox { color: red; } ");
        ui->tabWidget->tabBar()->setTabTextColor(1, Qt::red);
        ui->xsExplorePushButton->setEnabled(false);
    }

    if(m_quadLoaded)
    {
        //ui->quadGroupBox->setStyleSheet("QGroupBox { color: black; } ");
        ui->tabWidget->tabBar()->setTabTextColor(2, Qt::black);
        ui->quadExplorePushButton->setEnabled(true);
    }
    else
    {
        //ui->quadGroupBox->setStyleSheet("QGroupBox { color: red; } ");
        ui->quadExplorePushButton->setEnabled(false);
        ui->tabWidget->tabBar()->setTabTextColor(2, Qt::red);
    }

    if(m_solverLoaded)
    {
        //ui->paramsGroupBox->setStyleSheet("QGroupBox { color: black; } ");
        ui->tabWidget->tabBar()->setTabTextColor(3, Qt::black);
    }
    else
    {
        //ui->paramsGroupBox->setStyleSheet("QGroupBox { color: red; } ");
        ui->tabWidget->tabBar()->setTabTextColor(3, Qt::red);
    }

    if(m_srcLoaded)
    {
        ui->tabWidget->tabBar()->setTabTextColor(4, Qt::black);
    }
    else
    {
        ui->tabWidget->tabBar()->setTabTextColor(4, Qt::red);
    }

    //ui->geometryGroupBox->update();

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

void MainWindow::on_paramsTypeComboBox_activated(int indx)
{
    ui->paramsPnComboBox->setEnabled(indx > 1);
    m_solType = indx - 1; // Resetting to the default (indx = 0) rolls over to UINT_MAX

    m_solverLoaded = (indx == 1 || (indx > 1 && ui->paramsPnComboBox->currentIndex() > 1));

    updateLaunchButton();
}

void MainWindow::on_paramsPnComboBox_activated(int indx)
{
    m_pn = indx - 1;  // Resetting to the default rolls over to UINT_MAX

    m_solverLoaded = (ui->paramsTypeComboBox->currentIndex() == 1 || (ui->paramsTypeComboBox->currentIndex() > 1 && indx >= 1));

    updateLaunchButton();
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

    emit signalBeginXsParse(filename);
}

void MainWindow::on_xsExplorePushButton_clicked()
{
    xsDialog->setXs(m_parser);
    xsDialog->show();
}

void MainWindow::on_actionMCNP6_Generation_triggered()
{
    if(m_mesh == NULL)
    {
        QString errmsg = QString("The geometry must be loaded before a MCNP6 file can be generated.");
        QMessageBox::warning(this, "Insufficient Data", errmsg, QMessageBox::Close);
        return;
    }

    if(m_srcParams == NULL)
    {
        QString errmsg = QString("The energy spectra must be loaded before a MCNP6 file can be generated.");
        QMessageBox::warning(this, "Insufficient Data", errmsg, QMessageBox::Close);
        return;
    }

    if(!m_srcParams->update(energyDialog->getUserIntensity(), ui->sourceXDoubleSpinBox->value(), ui->sourceYDoubleSpinBox->value(), ui->sourceZDoubleSpinBox->value(),
                            ui->sourcePhiSpinBox->value(), ui->sourceThetaSpinBox->value(), ui->sourceNSpinBox->value(), ui->sourceDegreesCheckBox->isChecked(),
                            ui->sourceDistanceSpinBox->value(), ui->sourceWidthSpinBox->value(), ui->sourceHeightSpinBox->value(),
                            ui->sourceTypeComboBox->currentIndex()))
    {
        QString errmsg = QString("Could not update the energy intensity data. The energy structure may have changed.");
        QMessageBox::warning(this, "Invalid Vector Size", errmsg, QMessageBox::Close);
        return;
    }

    if(!m_srcParams->normalize())
    {
        QString errmsg = QString("The energy spectrum total is zero.");
        QMessageBox::warning(this, "Divide by zero", errmsg, QMessageBox::Close);
        return;
    }

    McnpWriter mcnpwriter;
    mcnpwriter.writeMcnp("../mcnp.gitignore/mcnp_out.inp", m_mesh, m_srcParams, false);
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

void MainWindow::xsParseFinished(AmpxParser *parser)
{
    m_xsLoaded = true;
    updateLaunchButton();
    outputDialog->setEnergyGroups(parser->getGammaEnergyGroups());
    energyDialog->setEnergy(parser);

    m_srcParams = new SourceParams(parser);

    ui->mainProgressBar->setValue(0);
}

bool MainWindow::buildMaterials(AmpxParser *parser)
{
    //qDebug() << "Generating materials";

    switch(m_mesh->material)
    {
    case MaterialUtils::HOUNSFIELD19:
        m_xs->setElements(MaterialUtils::hounsfieldRangePhantom19Elements, MaterialUtils::hounsfieldRangePhantom19Weights);
        break;
    case MaterialUtils::WATER:
        m_xs->setElements(MaterialUtils::waterElements, MaterialUtils::waterWeights);
        break;
    default:
        QString errmsg = QString("Material map ") + QString::number(m_mesh->material) + " was not found in the MaterialUtil database";
        QMessageBox::warning(NULL, "Internal Error", errmsg, QMessageBox::Close);
        return false;
    }

    if(!m_xs->allocateMemory(parser->getGammaEnergyGroups(), m_pn))
        return false;

    return m_xs->addAll(parser);

    // One extra material for the empty material at the end
    //m_xs->allocateMemory(static_cast<const unsigned int>(MaterialUtils::hounsfieldRangePhantom19.size()) + 1, parser->getGammaEnergyGroups(), m_pn);

    /*
    bool allPassed = true;

    // Add the materials to the xs library
    for(unsigned int i = 0; i < MaterialUtils::hounsfieldRangePhantom19.size(); i++)
        if(allPassed)
            allPassed &= m_xs->addMaterial(MaterialUtils::hounsfieldRangePhantom19Elements, MaterialUtils::hounsfieldRangePhantom19Weights[i], parser);

    // The last material is empty and should never be used
    if(allPassed)
        allPassed &= m_xs->addMaterial(std::vector<int>{}, std::vector<float>{}, parser);
        */

    //return allPassed;
}

void MainWindow::meshParseUpdateHandler(int x)
{
    qDebug() << "Here";
    ui->geomProgressBar->setValue(x);
}

void MainWindow::meshParseFinished(Mesh *mesh)
{
    m_mesh = mesh;

    if(m_mesh == NULL)
    {
        m_geomLoaded = false;
        updateLaunchButton();

        QString errmsg = QString("Could not load geometry. See debug log.");
        QMessageBox::warning(this, "Invalid Geometry", errmsg, QMessageBox::Close);

        return;
    }

    //OutWriter::writeArray("/media/Storage/thesis/mcnp.gitignore/ctdensity.dat", m_mesh->density);

    //qDebug() << "Here the zslice = " << m_mesh->zElemCt;
    geomDialog->updateMesh(m_mesh);
    outputDialog->updateMesh(m_mesh);
    m_geomLoaded = true;

    updateLaunchButton();

    ui->geomProgressBar->setValue(0);
}

void MainWindow::energyGroupsUpdated()
{
    if(m_srcParams != NULL)
    {
        if(!m_srcParams->update(energyDialog->getUserIntensity(), ui->sourceXDoubleSpinBox->value(), ui->sourceYDoubleSpinBox->value(), ui->sourceZDoubleSpinBox->value(),
                                ui->sourcePhiSpinBox->value(), ui->sourceThetaSpinBox->value(), ui->sourceNSpinBox->value(), ui->sourceDegreesCheckBox->isChecked(),
                                ui->sourceDistanceSpinBox->value(), ui->sourceWidthSpinBox->value(), ui->sourceHeightSpinBox->value(),
                                ui->sourceTypeComboBox->currentIndex()))
        {
            //qDebug() << "Whoops";
            return;
        }
        else
        {
            m_srcLoaded = true;
            updateLaunchButton();
        }
    }
    else
    {
        return;
        //qDebug() << "still null";
    }
}

void MainWindow::onRaytracerFinished(std::vector<RAY_T>* uncollided)
{
    m_raytrace = uncollided;

    switch(m_solType)
    {
    case MainWindow::ISOTROPIC:
        OutWriter::writeScalarFlux("raytrace_flux_iso.dat", *m_xs, *m_mesh, *uncollided);
        emit signalLaunchSolverIso(m_quad, m_mesh, m_xs, m_solvParams, m_srcParams, uncollided);
        break;
    case MainWindow::LEGENDRE:
        OutWriter::writeAngularFlux("raytrace_flux_leg.dat", *m_xs, *m_quad, *m_mesh, *uncollided);
        emit signalLaunchSolverLegendre(m_quad, m_mesh, m_xs, m_solvParams, m_srcParams, uncollided);
        break;
    case MainWindow::HARMONIC:
        OutWriter::writeAngularFlux("raytrace_flux_harm.dat", *m_xs, *m_quad, *m_mesh, *uncollided);
        emit signalLaunchSolverHarmonic(m_quad, m_mesh, m_xs, m_solvParams, m_srcParams, uncollided);
        break;
    default:
        qCritical() << "No solver of type " << m_solType;
    }
}

void MainWindow::onSolverFinished(std::vector<SOL_T> *solution)
{
    m_solution = solution;

    std::vector<SOL_T> scalar;

    switch(m_solType)
    {
    case MainWindow::ISOTROPIC:
        OutWriter::writeScalarFlux("solver_flux_iso.dat", *m_xs, *m_mesh, *solution);
        break;
    case MainWindow::LEGENDRE:
        OutWriter::writeAngularFlux("solver_angular_leg.dat", *m_xs, *m_quad, *m_mesh, *solution);
        scalar.resize(m_xs->groupCount() * m_mesh->voxelCount(), static_cast<SOL_T>(0));
        for(unsigned int ie = 0; ie < m_xs->groupCount(); ie++)
            for(unsigned int ir = 0; ir < m_mesh->voxelCount(); ir++)
            {
                unsigned int scalarOffset = ie*m_mesh->voxelCount() + ir;
                unsigned int angularOffset = ie*m_mesh->voxelCount()*m_quad->angleCount() + ir;
                for(unsigned int ia = 0; ia < m_quad->angleCount(); ia++)
                    scalar[scalarOffset] += (*solution)[angularOffset + ia*m_mesh->voxelCount()] * m_quad->wt[ia];
            }
        OutWriter::writeScalarFlux("solver_scalar_leg.dat", *m_xs, *m_mesh, scalar);

        break;
    case MainWindow::HARMONIC:
        OutWriter::writeAngularFlux("solver_flux_harm.dat", *m_xs, *m_quad, *m_mesh, *solution);
        break;
    default:
        qCritical() << "No solver of type " << m_solType;
    }
}

void MainWindow::on_solverGpuCheckBox_toggled(bool gpuOn)
{
    m_solvParams->gpu_accel = gpuOn;
}

void MainWindow::on_sourceTypeComboBox_activated(int v)
{

    QPixmap srcImg;
    switch(v)
    {
    case 0:
        ui->sourceImageLabel->setPixmapFile("/media/Storage/thesis/doctors/images/source1.png");  // Point source
        ui->sourcePhiSpinBox->setEnabled(false);
        ui->sourceThetaSpinBox->setEnabled(false);
        ui->sourceNSpinBox->setEnabled(false);
        break;
    case 1:
        ui->sourceImageLabel->setPixmapFile("/media/Storage/thesis/doctors/images/source2.png");  // Fan
        ui->sourcePhiSpinBox->setEnabled(true);
        ui->sourceThetaSpinBox->setEnabled(true);
        ui->sourceNSpinBox->setEnabled(false);
        break;
    case 2:
        ui->sourceImageLabel->setPixmapFile("/media/Storage/thesis/doctors/images/source3.png");  // Multifan
        ui->sourcePhiSpinBox->setEnabled(true);
        ui->sourceThetaSpinBox->setEnabled(true);
        ui->sourceNSpinBox->setEnabled(true);
        break;
    case 3:
        ui->sourceImageLabel->setPixmapFile("/media/Storage/thesis/doctors/images/source4.png"); // Cone
        ui->sourcePhiSpinBox->setEnabled(false);
        ui->sourceThetaSpinBox->setEnabled(true);
        ui->sourceNSpinBox->setEnabled(false);
        break;
    case 4:
        ui->sourceImageLabel->setPixmapFile("/media/Storage/thesis/doctors/images/source5.png"); // Multicone
        ui->sourcePhiSpinBox->setEnabled(false);
        ui->sourceThetaSpinBox->setEnabled(true);
        ui->sourceNSpinBox->setEnabled(true);
        break;
    default:
        qDebug() << "No picture associated with source type " << v;
    }
}
