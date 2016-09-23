#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QDebug>

#include <QMessageBox>
#include <iostream>
#include <cmath>

#include "treemodel.h"
#include "outwriter.h"
#include "legendre.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    parser(),
    model(NULL),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(ui->openButton, SIGNAL(pressed()), this, SLOT(openFile()));
    connect(ui->parseHeaderButton, SIGNAL(pressed()), this, SLOT(parseHeader()));
    connect(ui->parseDataButton, SIGNAL(pressed()), this, SLOT(parseData()));
    connect(&parser, SIGNAL(error(QString)), this, SLOT(displayError(QString)));

    model = new TreeModel(parser);
    ui->treeView->setModel(model);

    Legendre l;

    std::cout << l.getLegendrePoly(5) << std::endl;

    std::cout << "Done" << std::endl;

    qDebug() << doubleFactorial(8);

}

MainWindow::~MainWindow()
{
    delete ui;
    delete model;
}

void MainWindow::openFile()
{
    parser.openFile("/media/Storage/scale/scale6.2/data/scale.rev11.xn28g19v7.1");
    model->addFile("/media/Storage/scale/scale6.2/data/scale.rev11.xn28g19v7.1");

    //parser.openFile("/media/Storage/scale/Scale6.1/data/scale.rev10.xn27g19v7");
    //model->addFile("/media/Storage/scale/Scale6.1/data/scale.rev10.xn27g19v7");

    //parser.openFile("/media/Storage/scale/Scale6.1/data/scale.rev10.xn200g47v7");
    //model->addFile("/media/Storage/scale/Scale6.1/data/scale.rev10.xn200g47v7");

    ui->treeView->reset();
}

void MainWindow::parseHeader()
{
    parser.parseHeader();
    model->setupHeaderData();
    ui->treeView->reset();
    parser.debugAllEnergies();
}

void MainWindow::parseData()
{
    if(!parser.parseData())
        return;

    for(int i = 0; i < parser.getZaids().size(); i++)
    {

        NuclideData *nuc = parser.getData(i);

        const AmpxRecordParserType3 &nucdir = nuc->getDirectory();

        const AmpxRecordParserType10 &nucscatdir = nuc->getGammaScatterDirectory();

        const std::vector<int> &mts = nucscatdir.getMtList();

        bool foundit = false;
        for(unsigned int j = 0; j < mts.size(); j++)
            if(mts[j] == 502)
                foundit = true;

        if(!foundit)
            continue;

        //qDebug() << "Checking i = " << i;

        //for(unsigned int i = 0; i < mts.size(); i++)
        //    qDebug() << mts[i] << "  " << (be9scatdir.getNlList()[i] + 1);

        AmpxRecordParserType12 *nucscats = nuc->getGammaScatterMatrix(502, 0);
        float q = nucscats->getXs(18, 18);
        float r = nucscats->getXs(0, 0);

        //if(q > 0.2257/(4*M_PI) && q < 0.2258/(4*M_PI))
        if(q > 0.2257 && q < 0.2258)
        {
            qDebug() << "Found a match! index: " << i;
            qDebug() << nucdir.getText();
        }
        qDebug() << i << " " << q << " " << r << " " << q/r;

        const AmpxRecordParserType9 &gxs = nuc->getGammaXs();
        int mtTotIndx = gxs.getMtIndex(501);
        std::vector<float> tot = gxs.getSigmaMt(mtTotIndx);

        float s = tot[0];
        float t = tot[18];

        //qDebug() << i << " " << s << " " << t << " " << s/t << nucdir.getText();

        if(s > 0.0415 && s < 0.0416)
        {
            qDebug() << "Found a match in s! index: " << i;
            qDebug() << nucdir.getText();
        }

        if(t > 0.0415 && t < 0.0416)
        {
            qDebug() << "Found a match in t! index: " << i;
            qDebug() << nucdir.getText();
        }

        if(i == 57)
            for(int ii = 0; ii < tot.size(); ii++)
                qDebug() << tot[ii];

        //qDebug() << "Total scatter matrices in Be9: " << nucscats.size();

        //AmpxRecordParserType12 *be9scat = nuc->getGammaScatterMatrix(502, 0);

        //OutWriter o("/media/Storage/thesis/python/xsdataPlotter/be9scatter504.dat");
        //o.writeGammaScatterMatrix(parser.getGammaEnergy(), nuc, 502);
    }

    qDebug() << "Finished searching";



    //int be9indx = parser.findIndexByZaid(4009);
    //qDebug() << "Be index: " << be9indx;

    NuclideData *be9 = parser.getData(58);

    const AmpxRecordParserType3 &be9dir = be9->getDirectory();

    qDebug() << "Be9 text: " << be9dir.getText();

    int procs = be9dir.getAveragedGammaProcCount();
    qDebug() << "Total averaged gamma processes: " << procs;

    const AmpxRecordParserType9 &gxs = be9->getGammaXs();

    qDebug() << gxs.getMtCount() << " MT values: ";
    for(int i = 0; i < gxs.getMtCount(); i++)
        qDebug() << gxs.getMtList()[i];

    int mtTotIndx = gxs.getMtIndex(501);
    int mtPairIndx = gxs.getMtIndex(516);
    int mtCohIndx = gxs.getMtIndex(502);
    int mtIncIndx = gxs.getMtIndex(504);
    int mtPeIndx = gxs.getMtIndex(522);

    std::vector<float> tot = gxs.getSigmaMt(mtTotIndx);
    std::vector<float> pair = gxs.getSigmaMt(mtPairIndx);
    std::vector<float> coh = gxs.getSigmaMt(mtCohIndx);
    std::vector<float> inc = gxs.getSigmaMt(mtIncIndx);
    std::vector<float> pe = gxs.getSigmaMt(mtPeIndx);

    qDebug() << "SCATTER DATA";
    const AmpxRecordParserType10 &be9scatdir = be9->getGammaScatterDirectory();

    const std::vector<int> &mts = be9scatdir.getMtList();

    for(unsigned int i = 0; i < mts.size(); i++)
        qDebug() << mts[i] << "  " << (be9scatdir.getNlList()[i] + 1);



    const std::vector<AmpxRecordParserType12*> &be9scats = be9->getGammaScatterMatrices();

    qDebug() << "Total scatter matrices in Be9: " << be9scats.size();

    //AmpxRecordParserType12 *be9scat = be9->getGammaScatterMatrix(504, 6);

    OutWriter o("/media/Storage/thesis/python/xsdataPlotter/cscatter502.dat");
    o.writeGammaScatterMatrix(parser.getGammaEnergy(), be9, 504);


}

void MainWindow::displayError(QString msg)
{
    QMessageBox::critical(this, "Error", msg);
}
