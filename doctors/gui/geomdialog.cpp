#include "geomdialog.h"
#include "ui_geomdialog.h"

#include <QDebug>

#include "mesh.h"

GeomDialog::GeomDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::GeomDialog),
    m_mesh(NULL),
    m_rendertype(0)
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

    connect(ui->materialRadioButton, SIGNAL(clicked()), this, SLOT(updateRenderType()));
    connect(ui->densityRadioButton, SIGNAL(clicked()), this, SLOT(updateRenderType()));
    connect(ui->atomDensityRadioButton, SIGNAL(clicked()), this, SLOT(updateRenderType()));
    connect(ui->ctRadioButton, SIGNAL(clicked()), this, SLOT(updateRenderType()));

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

// Negative level will not change level
void GeomDialog::setSliceLevel(int level)
{
    qDebug() << "Set the slice to " << level;

    if(m_mesh == NULL)
    {
        qDebug() << "ERROR: setSliceLevel on a NULL pointer";
        return;
    }

    if(level < 0)
        level = ui->sliceSpinBox->value();

    //loadUniqueBrush();
    //loadParulaBrush();
    //loadViridis256Brush();
    //loadPhantom19Brush();

    if(m_rendertype == 0)
        loadPhantom19Brush();
    else if(m_rendertype == 1)
        loadViridis256Brush();
    else if(m_rendertype == 2)
        loadViridis256Brush();
    else if(m_rendertype == 3)
        loadViridis256Brush();
    else
    {
        qDebug() << "Geomdialog::setSliceLevel(): 87: rendertype was illegal";
        return;
    }

    int zid;

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
                if(m_rendertype == 0)
                {
                    zid = m_mesh->zoneId[i*m_mesh->yElemCt*m_mesh->zElemCt + j*m_mesh->zElemCt + level];
                }
                else if(m_rendertype == 1)
                {
                    zid = m_mesh->density[i*m_mesh->yElemCt*m_mesh->zElemCt + j*m_mesh->zElemCt + level] * 200;
                    if(zid > 255)
                        zid = 255;
                    if(zid < 0)
                        zid = 0;
                }
                else if(m_rendertype == 3)
                {
                    zid = m_mesh->atomDensity[i*m_mesh->yElemCt*m_mesh->zElemCt + j*m_mesh->zElemCt + level];
                    if(zid > 255)
                        zid = 255;
                    if(zid < 0)
                        zid = 0;
                }
                else if(m_rendertype == 2)
                {
                    zid = (m_mesh->ct[i*m_mesh->yElemCt*m_mesh->zElemCt + j*m_mesh->zElemCt + level] + 1024) / 12.0;
                    if(zid > 255)
                        zid = 255;
                }
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

        for(unsigned int i = 0; i < m_mesh->xElemCt; i++)
            for(unsigned int j = 0; j < m_mesh->zElemCt; j++)
            {
                if(m_rendertype == 0)
                {
                    zid = m_mesh->zoneId[i*m_mesh->yElemCt*m_mesh->zElemCt + level*m_mesh->zElemCt + j];
                }
                else if(m_rendertype == 1)
                {
                    zid = m_mesh->density[i*m_mesh->yElemCt*m_mesh->zElemCt + level*m_mesh->zElemCt + j] * 200;
                    if(zid > 255)
                        zid = 255;
                    if(zid < 0)
                        zid = 0;
                }
                else if(m_rendertype == 3)
                {
                    zid = m_mesh->atomDensity[i*m_mesh->yElemCt*m_mesh->zElemCt + level*m_mesh->zElemCt + j];
                    if(zid > 255)
                        zid = 255;
                    if(zid < 0)
                        zid = 0;
                }
                else if(m_rendertype == 2)
                {
                    zid = (m_mesh->ct[i*m_mesh->yElemCt*m_mesh->zElemCt + level*m_mesh->zElemCt + j] + 1024) / 12.0;
                    if(zid > 255)
                        zid = 255;
                }
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
                if(m_rendertype == 0)
                {
                    zid = m_mesh->zoneId[level*m_mesh->yElemCt*m_mesh->zElemCt + i*m_mesh->zElemCt + j];
                }
                else if(m_rendertype == 1)
                {
                    zid = m_mesh->density[level*m_mesh->yElemCt*m_mesh->zElemCt + i*m_mesh->zElemCt + j] * 200;
                    if(zid > 255)
                        zid = 255;
                    if(zid < 0)
                        zid = 0;
                }
                else if(m_rendertype == 3)
                {
                    zid = m_mesh->atomDensity[level*m_mesh->yElemCt*m_mesh->zElemCt + i*m_mesh->zElemCt + j];
                    if(zid > 255)
                        zid = 255;
                    if(zid < 0)
                        zid = 0;
                }
                else if(m_rendertype == 2)
                {
                    zid = (m_mesh->ct[level*m_mesh->yElemCt*m_mesh->zElemCt + i*m_mesh->zElemCt + j] + 1024) / 12.0;
                    if(zid > 255)
                        zid = 255;
                }
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


void GeomDialog::updateRenderType()
{
    if(ui->materialRadioButton->isChecked())
    {
        m_rendertype = 0;
        setSliceLevel(-1);
    }
    else if(ui->densityRadioButton->isChecked())
    {
        m_rendertype = 1;
        setSliceLevel(-1);
    }
    else if(ui->atomDensityRadioButton->isChecked())
    {
        m_rendertype = 3;
        setSliceLevel(-1);
    }
    else if(ui->ctRadioButton->isChecked())
    {
        m_rendertype = 2;
        setSliceLevel(-1);
    }
}












