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

    loadParula();

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

void OutputDialog::loadParula()
{
    brushes.clear();

    brushes.push_back(QBrush(QColor::fromRgbF(0.2081,    0.1663,    0.5292)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.2116,    0.1898,    0.5777)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.2123,    0.2138,    0.6270)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.2081,    0.2386,    0.6771)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.1959,    0.2645,    0.7279)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.1707,    0.2919,    0.7792)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.1253,    0.3242,    0.8303)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0591,    0.3598,    0.8683)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0117,    0.3875,    0.8820)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0060,    0.4086,    0.8828)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0165,    0.4266,    0.8786)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0329,    0.4430,    0.8720)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0498,    0.4586,    0.8641)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0629,    0.4737,    0.8554)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0723,    0.4887,    0.8467)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0779,    0.5040,    0.8384)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0793,    0.5200,    0.8312)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0749,    0.5375,    0.8263)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0641,    0.5570,    0.8240)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0488,    0.5772,    0.8228)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0343,    0.5966,    0.8199)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0265,    0.6137,    0.8135)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0239,    0.6287,    0.8038)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0231,    0.6418,    0.7913)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0228,    0.6535,    0.7768)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0267,    0.6642,    0.7607)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0384,    0.6743,    0.7436)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0590,    0.6838,    0.7254)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0843,    0.6928,    0.7062)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.1133,    0.7015,    0.6859)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.1453,    0.7098,    0.6646)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.1801,    0.7177,    0.6424)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.2178,    0.7250,    0.6193)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.2586,    0.7317,    0.5954)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.3022,    0.7376,    0.5712)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.3482,    0.7424,    0.5473)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.3953,    0.7459,    0.5244)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.4420,    0.7481,    0.5033)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.4871,    0.7491,    0.4840)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.5300,    0.7491,    0.4661)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.5709,    0.7485,    0.4494)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.6099,    0.7473,    0.4337)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.6473,    0.7456,    0.4188)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.6834,    0.7435,    0.4044)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.7184,    0.7411,    0.3905)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.7525,    0.7384,    0.3768)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.7858,    0.7356,    0.3633)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.8185,    0.7327,    0.3498)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.8507,    0.7299,    0.3360)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.8824,    0.7274,    0.3217)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9139,    0.7258,    0.3063)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9450,    0.7261,    0.2886)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9739,    0.7314,    0.2666)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9938,    0.7455,    0.2403)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9990,    0.7653,    0.2164)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9955,    0.7861,    0.1967)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9880,    0.8066,    0.1794)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9789,    0.8271,    0.1633)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9697,    0.8481,    0.1475)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9626,    0.8705,    0.1309)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9589,    0.8949,    0.1132)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9598,    0.9218,    0.0948)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9661,    0.9514,    0.0755)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9763,    0.9831,    0.0538)));
}
