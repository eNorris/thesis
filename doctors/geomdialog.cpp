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

    //connect(ui->buttonBox, SIGNAL(accepted()), this, SLOT(updateMeshSlicePlane()));
    connect(ui->xyRadioButton, SIGNAL(clicked()), this, SLOT(updateMeshSlicePlane()));
    connect(ui->xzRadioButton, SIGNAL(clicked()), this, SLOT(updateMeshSlicePlane()));
    connect(ui->yzRadioButton, SIGNAL(clicked()), this, SLOT(updateMeshSlicePlane()));

    //connect(ui->xyRadioButton, SIGNAL())

    // TODO updateMeshSlicePlane when a radio button is selected
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

    updateMeshSlicePlane();
}

void GeomDialog::setSliceLevel(int level)
{
    qDebug() << "Set the slice to " << level;

    if(m_mesh == NULL)
    {
        qDebug() << "ERROR: setSliceLevel on a NULL pointer";
        return;
    }

    loadUniqueBrush();

    if(ui->xyRadioButton->isChecked())
    {
        if(level >= m_mesh->zMesh)
        {
            qDebug() << "level is too high! z-slices = " << m_mesh->zMesh;
            return;
        }


        for(int i = 0; i < m_mesh->xMesh; i++)
            for(int j = 0; j < m_mesh->yMesh; j++)
            {
                int zid = m_mesh->zoneId[i*m_mesh->yMesh*m_mesh->zMesh + j*m_mesh->zMesh + level];
                //int rindx = i*m_mesh->xMesh + j;
                rects[i*m_mesh->yMesh + j]->setBrush(brushes[zid]);
            }
    }
    else if(ui->xzRadioButton->isChecked())
    {
        if(level >= m_mesh->yMesh)
        {
            qDebug() << "level is too high! y-slices = " << m_mesh->yMesh;
            return;
        }

        //loadUniqueBrush();
        for(int i = 0; i < m_mesh->xMesh; i++)
            for(int j = 0; j < m_mesh->zMesh; j++)
            {
                int zid = m_mesh->zoneId[i*m_mesh->yMesh*m_mesh->zMesh + level*m_mesh->zMesh + j];
                rects[i*m_mesh->zMesh + j]->setBrush(brushes[zid]);
            }
    }
    else if(ui->yzRadioButton->isChecked())
    {
        if(level >= m_mesh->xMesh)
        {
            qDebug() << "level is too high! x-slices = " << m_mesh->xMesh;
            return;
        }

        //loadUniqueBrush();
        for(int i = 0; i < m_mesh->yMesh; i++)
            for(int j = 0; j < m_mesh->zMesh; j++)
            {
                int zid = m_mesh->zoneId[level*m_mesh->yMesh*m_mesh->zMesh + i*m_mesh->zMesh + j];
                rects[i*m_mesh->zMesh + j]->setBrush(brushes[zid]);
            }
    }
    else
    {
        qDebug() << "Updating mesh colorer, but no slice plane was selected!";
        return;
    }

}

void GeomDialog::updateMeshSlicePlane()
{
    rects.clear();
    scene->clear();

    if(ui->xyRadioButton->isChecked())
    {
        qDebug() << "Setting to XY slice";


        QBrush greenBrush(Qt::green);
        for(int i = 0; i < m_mesh->xMesh; i++)
            for(int j = 0; j < m_mesh->yMesh; j++)
            {
                //qreal x = m_mesh->xIndex[i];
                //qreal y = m_mesh->yIndex[j];
                //qDebug() << "x,y = " << x << ", " << y;
                rects.push_back(scene->addRect(m_mesh->xIndex[i], m_mesh->yIndex[j], m_mesh->dx[i], m_mesh->dy[j], Qt::NoPen, greenBrush));
            }

        ui->sliceSpinBox->setMaximum(m_mesh->zMesh-1);
        ui->sliceVerticalSlider->setMaximum(m_mesh->zMesh-1);
    }
    else if(ui->xzRadioButton->isChecked())
    {
        qDebug() << "XZ max = " << m_mesh->yMesh-1;


        QBrush greenBrush(Qt::green);
        for(int i = 0; i < m_mesh->xMesh; i++)
            for(int j = 0; j < m_mesh->zMesh; j++)
            {
                //qreal x = m_mesh->xIndex[i];
                //qreal z = m_mesh->zIndex[j];
                //qDebug() << "x,y = " << x << ", " << z;
                rects.push_back(scene->addRect(m_mesh->xIndex[i], m_mesh->zIndex[j], m_mesh->dx[i], m_mesh->dz[j], Qt::NoPen, greenBrush));
            }

        ui->sliceSpinBox->setMaximum(m_mesh->yMesh-1);
        ui->sliceVerticalSlider->setMaximum(m_mesh->yMesh-1);
    }
    else if(ui->yzRadioButton->isChecked())
    {
        qDebug() << "Setting to YZ slice";


        QBrush greenBrush(Qt::green);
        for(int i = 0; i < m_mesh->yMesh; i++)
            for(int j = 0; j < m_mesh->zMesh; j++)
            {
                //qreal y = m_mesh->yIndex[i];
                //qreal z = m_mesh->zIndex[j];
                //qDebug() << "x,y = " << y << ", " << z;
                rects.push_back(scene->addRect(m_mesh->yIndex[i], m_mesh->zIndex[j], m_mesh->dy[i], m_mesh->dz[j], Qt::NoPen, greenBrush));
            }

        ui->sliceSpinBox->setMaximum(m_mesh->xMesh-1);
        ui->sliceVerticalSlider->setMaximum(m_mesh->xMesh-1);
    }
    else
    {
        qDebug() << "Updating mesh renderer, but no slice plane was selected!";
        return;
    }

    ui->sliceSpinBox->setValue(0);
    ui->sliceVerticalSlider->setValue(0);
    setSliceLevel(0);
}















