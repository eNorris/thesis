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

    QBrush greenBrush(Qt::green);

    QPen outlinePen(Qt::black);
    outlinePen.setWidth(2);

    rect = scene->addRect(100, 0, 80, 100, outlinePen, greenBrush);
}

OutputDialog::~OutputDialog()
{
    delete ui;
}

void OutputDialog::disp(std::vector<float>)
{
    qDebug() << "Displaying a new vector...";
}
