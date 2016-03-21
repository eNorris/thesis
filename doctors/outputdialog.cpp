#include "outputdialog.h"
#include "ui_outputdialog.h"

#include <QDebug>
#include <cmath>

#include "mesh.h"

OutputDialog::OutputDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::OutputDialog)
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
}

OutputDialog::~OutputDialog()
{
    delete ui;
}

void OutputDialog::updateMesh(Mesh *mesh)
{
    m_mesh = mesh;

    // I don't know the geometry anymore, so it doesn't make sense to draw anything
    //updateMeshSlicePlane();
}

void OutputDialog::updateSolution(std::vector<float> data)
{
    m_data = data;
}


void OutputDialog::reRender(std::vector<float> data)
{
    updateSolution(data);
    updateMeshSlicePlane();
    setSliceLevel(ui->sliceVerticalSlider->value());
}


void OutputDialog::setSliceLevel(int level)
{
    //qDebug() << "Set the slice to " << level;

    if(m_mesh == NULL)
    {
        qDebug() << "ERROR: setSliceLevel on a NULL pointer!";
        return;
    }

    if(m_data.size() == 0)
    {
        qDebug() << "ERROR: Setting level slice with no solution data";
        return;
    }

    if(rects.size() == 0)
    {
        qDebug() << "ERROR: There are no rectangles in the drawing pipeline!";
        qDebug() << "Did you call the mesher?";
        return;
    }

    loadParulaBrush();

    float minval = 1E35;
    float maxval = -1E35;

    if(ui->xyRadioButton->isChecked())
    {
        if(level >= m_mesh->zMesh)
        {
            qDebug() << "level is too high! z-slices = " << m_mesh->zMesh;
            return;
        }

        for(int ix = 0; ix < m_mesh->xMesh; ix++)
            for(int iy = 0; iy < m_mesh->yMesh; iy++)
            {
                if(m_data[ix*m_mesh->yMesh*m_mesh->zMesh + iy*m_mesh->zMesh + level] < minval)
                    minval = m_data[ix*m_mesh->yMesh*m_mesh->zMesh + iy*m_mesh->zMesh + level];
                else if(m_data[ix*m_mesh->yMesh*m_mesh->zMesh + iy*m_mesh->zMesh + level] > maxval)
                    maxval = m_data[ix*m_mesh->yMesh*m_mesh->zMesh + iy*m_mesh->zMesh + level];
            }

        if(maxval == 0)
        {
            qDebug() << "Zero flux everywhere!";
            return;
        }

        if(minval < 0)
        {
            qDebug() << "WARNING: Negative flux!";
        }

        if((maxval - minval) / maxval < 1E-5)
        {
            qDebug() << "Displaying a flat surface!";
            return;
        }

        for(int i = 0; i < m_mesh->xMesh; i++)
            for(int j = 0; j < m_mesh->yMesh; j++)
            {
                //int zid = m_mesh->zoneId[i*m_mesh->yMesh*m_mesh->zMesh + j*m_mesh->zMesh + level];
                float flux = m_data[i*m_mesh->yMesh*m_mesh->zMesh + j*m_mesh->zMesh + level];
                int fid = round(63*(flux-minval) / (maxval-minval));
                rects[i*m_mesh->yMesh + j]->setBrush(brushes[fid]);
            }
    }
    else if(ui->xzRadioButton->isChecked())
    {
        if(level >= m_mesh->yMesh)
        {
            qDebug() << "level is too high! y-slices = " << m_mesh->yMesh;
            return;
        }

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

void OutputDialog::updateMeshSlicePlane()
{
    qDebug() << "Calling the mesher!";

    rects.clear();
    scene->clear();

    if(ui->xyRadioButton->isChecked())
    {
        qDebug() << "Setting to XY slice";

        QBrush greenBrush(Qt::green);
        for(int i = 0; i < m_mesh->xMesh; i++)
            for(int j = 0; j < m_mesh->yMesh; j++)
            {
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



























