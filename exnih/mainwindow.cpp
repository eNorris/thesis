#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "AmpxLibrary.h"
#include "AmpxReader.h"

#include <fstream>
#include <QTime>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    AmpxLibrary lib;
    AmpxReader reader;

    std::fstream fin;
    //fin.open("/media/Storage/scale/scale6.2/data/scale.rev11.xn28g19v7.1");
    fin.open("/media/Storage/scale/Scale6.1/data/scale.rev10.xn27g19v7");
    //fin.open("/media/Storage/scale/Scale6.1/data/scale.rev10.xn200g47v7");

    QTime t;
    t.start();
    reader.read(&fin, lib, true, false);
    int millis = t.elapsed();

    QList<LibraryNuclide*> nucs = lib.getConstNuclides();

    LibraryNuclide *nuc = nucs[11];

    qDebug() << "Nuclide: " << qPrintable(nuc->getDescription());

    qDebug() << "Number of 2d datasets: " << nuc->getGamma2dList().size();

    CrossSection2d *xs = nuc->getGamma2dDataByMt(504);

    qDebug() << "Legendre Order for MT=504: " << xs->getLegendreOrder();

    qDebug() << "Temperatures: " << xs->getNumberTemperatures();
    for(int i = 0; i < xs->getNumberTemperatures(); i++)
        qDebug() << "T: " << xs->getTemperatureList()[i];

    QList<ScatterMatrix*> *scats = xs->getScatterMatrices();

    qDebug() << "Size of scats: " << scats->size();

    ScatterMatrix *scat = (*scats)[0];
    SinkGroup *g = scat->getSinkGroup(0);

    float x = g->get(19);

    qDebug() << "Size of sink: " << (g->getStart() - g->getEnd());
    qDebug() << "P(0,0) = " << x;

    for(int snk = 1; snk <= 19; snk++)
    {
        SinkGroup *s = scat->getSinkGroupByGrp(snk);
        qDebug() << "Sinks: " << s->getStart() << " - " << s->getEnd();
        for(int src = 0; src < 19; src++)
        {
            //if(src > s->getEnd())
            //    std::cout << "0\t";
            //else if(src < s->getStart())
            //    std::cout << "0\t";
            //else
            std::cout << s->get(src) << "\t";
        }
        std::cout << std::endl;
    }

    qDebug() << "Millis: " << millis;

    //int read(fstream * file, AmpxLibrary & library, bool printErrors = true, bool verbose=false);
}

MainWindow::~MainWindow()
{
    delete ui;
}
