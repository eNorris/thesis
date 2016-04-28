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
    ui->graphicsView->setMouseTracking(true);
    //ui->graphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    //ui->graphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);


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
        if(level >= (signed) m_mesh->zElemCt)
        {
            qDebug() << "level is too high! z-slices = " << m_mesh->zElemCt;
            return;
        }


        for(unsigned int i = 0; i < m_mesh->xElemCt; i++)
            for(unsigned int j = 0; j < m_mesh->yElemCt; j++)
            {
                int zid = m_mesh->zoneId[i*m_mesh->yElemCt*m_mesh->zElemCt + j*m_mesh->zElemCt + level];
                //int rindx = i*m_mesh->xMesh + j;
                rects[i*m_mesh->yElemCt + j]->setBrush(brushes[zid]);
            }
    }
    else if(ui->xzRadioButton->isChecked())
    {
        if(level >= (signed) m_mesh->yElemCt)
        {
            qDebug() << "level is too high! y-slices = " << m_mesh->yElemCt;
            return;
        }

        //loadUniqueBrush();
        for(unsigned int i = 0; i < m_mesh->xElemCt; i++)
            for(unsigned int j = 0; j < m_mesh->zElemCt; j++)
            {
                int zid = m_mesh->zoneId[i*m_mesh->yElemCt*m_mesh->zElemCt + level*m_mesh->zElemCt + j];
                rects[i*m_mesh->zElemCt + j]->setBrush(brushes[zid]);
            }
    }
    else if(ui->yzRadioButton->isChecked())
    {
        if(level >= (signed) m_mesh->xElemCt)
        {
            qDebug() << "level is too high! x-slices = " << m_mesh->xElemCt;
            return;
        }

        //loadUniqueBrush();
        for(unsigned int i = 0; i < m_mesh->yElemCt; i++)
            for(unsigned int j = 0; j < m_mesh->zElemCt; j++)
            {
                int zid = m_mesh->zoneId[level*m_mesh->yElemCt*m_mesh->zElemCt + i*m_mesh->zElemCt + j];
                rects[i*m_mesh->zElemCt + j]->setBrush(brushes[zid]);
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
        for(unsigned int i = 0; i < m_mesh->xElemCt; i++)
            for(unsigned int j = 0; j < m_mesh->yElemCt; j++)
            {
                //qreal x = m_mesh->xIndex[i];
                //qreal y = m_mesh->yIndex[j];
                //qDebug() << "x,y = " << x << ", " << y;
                rects.push_back(scene->addRect(m_mesh->xNodes[i], m_mesh->yNodes[j], m_mesh->dx[i], m_mesh->dy[j], Qt::NoPen, greenBrush));
            }

        ui->sliceSpinBox->setMaximum(m_mesh->zElemCt-1);
        ui->sliceVerticalSlider->setMaximum(m_mesh->zElemCt-1);
    }
    else if(ui->xzRadioButton->isChecked())
    {
        qDebug() << "XZ max = " << m_mesh->yElemCt-1;


        QBrush greenBrush(Qt::green);
        for(unsigned int i = 0; i < m_mesh->xElemCt; i++)
            for(unsigned int j = 0; j < m_mesh->zElemCt; j++)
            {
                //qreal x = m_mesh->xIndex[i];
                //qreal z = m_mesh->zIndex[j];
                //qDebug() << "x,y = " << x << ", " << z;
                rects.push_back(scene->addRect(m_mesh->xNodes[i], m_mesh->zNodes[j], m_mesh->dx[i], m_mesh->dz[j], Qt::NoPen, greenBrush));
            }

        ui->sliceSpinBox->setMaximum(m_mesh->yElemCt-1);
        ui->sliceVerticalSlider->setMaximum(m_mesh->yElemCt-1);
    }
    else if(ui->yzRadioButton->isChecked())
    {
        qDebug() << "Setting to YZ slice";


        QBrush greenBrush(Qt::green);
        for(unsigned int i = 0; i < m_mesh->yElemCt; i++)
            for(unsigned int j = 0; j < m_mesh->zElemCt; j++)
            {
                //qreal y = m_mesh->yIndex[i];
                //qreal z = m_mesh->zIndex[j];
                //qDebug() << "x,y = " << y << ", " << z;
                rects.push_back(scene->addRect(m_mesh->yNodes[i], m_mesh->zNodes[j], m_mesh->dy[i], m_mesh->dz[j], Qt::NoPen, greenBrush));
            }

        ui->sliceSpinBox->setMaximum(m_mesh->xElemCt-1);
        ui->sliceVerticalSlider->setMaximum(m_mesh->xElemCt-1);
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















