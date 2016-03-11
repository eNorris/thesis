#include "geomdialog.h"
#include "ui_geomdialog.h"

#include <QDebug>

#include "mesh.h"

GeomDialog::GeomDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::GeomDialog),
    m_mesh(NULL)
{
    ui->setupUi(this);

    scene = new QGraphicsScene(this);
    ui->graphicsView->setScene(scene);

    ui->graphicsView->setDragMode(QGraphicsView::ScrollHandDrag);
    ui->graphicsView->setInteractive(true);

    connect(ui->sliceVerticalSlider, SIGNAL(sliderMoved(int)), ui->sliceSpinBox, SLOT(setValue(int)));
    connect(ui->sliceSpinBox, SIGNAL(valueChanged(int)), ui->sliceVerticalSlider, SLOT(setValue(int)));

    connect(ui->sliceSpinBox, SIGNAL(valueChanged(int)), this, SLOT(setSliceLevel(int)));
}

GeomDialog::~GeomDialog()
{
    delete ui;

    if(m_mesh != NULL)
        delete m_mesh;
}

void GeomDialog::updateMesh(Mesh *mesh)
{
    m_mesh = mesh;

    //qDebug() << "DEBUG: mesh zslice = " << m_mesh->

    if(ui->xyRadioButton->isEnabled())
    {
        rects.clear();
        scene->clear();

        ui->sliceSpinBox->setMaximum(m_mesh->zMesh-1);
        ui->sliceVerticalSlider->setMaximum(m_mesh->zMesh-1);

        //for(int i = 0; i < m_mesh->xIndex * m_mesh->yMesh; i++)
        //    rects.push_back(scene->addRect(5*i, 5*j, 5, 5, Qt::NoPen, greenBrush));
        QBrush greenBrush(Qt::green);
        for(int i = 0; i < m_mesh->xMesh; i++)
            for(int j = 0; j < m_mesh->yMesh; j++)
                rects.push_back(scene->addRect(m_mesh->xIndex[i], m_mesh->yIndex[j], m_mesh->dx[i], m_mesh->dy[j], Qt::NoPen, greenBrush));

        //for(int i = 0; i < 100*100; i++)
        //    rects[i]->setBrush(brushes[i % brushes.size()]);

        //for(int i = 0; i < 100; i++)
        //    for(int j = 0; j < 100; j++)
        //        rects.push_back(scene->addRect(5*i, 5*j, 5, 5, Qt::NoPen, greenBrush));
    }
    else if(ui->xzRadioButton->isEnabled())
    {

    }
    else if(ui->yzRadioButton->isEnabled())
    {

    }
    else
    {
        qDebug() << "Updating mesh renderer, but no slice plane was selected!";
        return;
    }
}

void GeomDialog::setSliceLevel(int level)
{
    qDebug() << "Set the slice to " << level;

    if(m_mesh == NULL)
    {
        qDebug() << "ERROR: setSliceLevel on a NULL pointer";
        return;
    }

    if(ui->xyRadioButton->isEnabled())
    {
        //rects.clear();
        //scene->clear();

        if(level >= m_mesh->zMesh)
        {
            qDebug() << "level is too high! z-slices = " << m_mesh->zMesh;
            return;
        }

        //for(int i = 0; i < m_mesh->xIndex * m_mesh->yMesh; i++)
        //    rects.push_back(scene->addRect(5*i, 5*j, 5, 5, Qt::NoPen, greenBrush));
        //QBrush greenBrush(Qt::green);
        loadUniqueBrush();
        for(int i = 0; i < m_mesh->xMesh; i++)
            for(int j = 0; j < m_mesh->yMesh; j++)
            {
                int zid = m_mesh->zoneId[i*m_mesh->yMesh*m_mesh->zMesh + j*m_mesh->zMesh + level];
                rects[i*m_mesh->yMesh + j]->setBrush(brushes[zid]);
            }
                //rects.push_back(scene->addRect(m_mesh->xIndex[i], m_mesh->yIndex[j], m_mesh->dx[i], m_mesh->dy[j], Qt::NoPen, greenBrush));

        //for(int i = 0; i < 100*100; i++)
        //    rects[i]->setBrush(brushes[i % brushes.size()]);

        //for(int i = 0; i < 100; i++)
        //    for(int j = 0; j < 100; j++)
        //        rects.push_back(scene->addRect(5*i, 5*j, 5, 5, Qt::NoPen, greenBrush));
    }
    else if(ui->xzRadioButton->isEnabled())
    {

    }
    else if(ui->yzRadioButton->isEnabled())
    {

    }
    else
    {
        qDebug() << "Updating mesh colorer, but no slice plane was selected!";
        return;
    }

}

















