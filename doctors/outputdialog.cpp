#include "outputdialog.h"
#include "ui_outputdialog.h"

#include <QDebug>

OutputDialog::OutputDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::OutputDialog)
{
    ui->setupUi(this);

    scene = new QGraphicsScene(this);
    ui->graphicsView->setScene(scene);

    ui->graphicsView->setDragMode(QGraphicsView::ScrollHandDrag);
    ui->graphicsView->setInteractive(true);


    QBrush greenBrush(Qt::green);

    //for(int i = 0; i < 64; i++)
    //{
    //    qDebug() << i*255.0/64;
    //    QBrush b(QColor(i*255.0/64, 50, 50));
    //    brushes.push_back(b);
    //}

    loadParulaBrush();

    for(int i = 0; i < 100; i++)
        for(int j = 0; j < 100; j++)
            rects.push_back(scene->addRect(5*i, 5*j, 5, 5, Qt::NoPen, greenBrush));

    for(int i = 0; i < 100*100; i++)
        rects[i]->setBrush(brushes[i % brushes.size()]);


    QPen outlinePen(Qt::black);
    outlinePen.setWidth(2);

    //rect = scene->addRect(100, 0, 80, 100, outlinePen, greenBrush);
}

OutputDialog::~OutputDialog()
{
    delete ui;
}

void OutputDialog::disp(std::vector<float> vdata)
{
    qDebug() << "Displaying a new vector...";

    float minval = 1E35;
    float maxval = -1E35;

    for(int i = 0; i < vdata.size(); i++)
    {
        if(vdata[i] < minval)
            minval = vdata[i];
        else if(vdata[i] > maxval)
            maxval = vdata[i];
    }

    if((maxval - minval) / maxval < 1E-5)
    {
        qDebug() << "Displaying a flat surface!";
        return;
    }

    if(rects.size() != vdata.size())
    {
        qDebug() << "Displaying a mismatch of data!";
        return;
    }

    for(int i = 0; i < rects.size(); i++)
        rects[i]->setBrush(brushes[((maxval-vdata[i])/(vdata[i]-minval) + 1)*63]);
}


