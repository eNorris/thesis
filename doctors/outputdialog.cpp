#include "outputdialog.h"
#include "ui_outputdialog.h"

#include <QDebug>
#include <cmath>
#include <QString>
#include <QStringListModel>

#include "mainwindow.h"
#include "mesh.h"

OutputDialog::OutputDialog(QWidget *parent) :
    QDialog(parent),
    m_logInterp(false),
    ui(new Ui::OutputDialog),
    m_listModel(NULL)
{
    ui->setupUi(this);

    m_listModel = new QStringListModel(this);
    ui->listView->setModel(m_listModel);

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

    connect(ui->linearInterpRadioButton, SIGNAL(clicked()), this, SLOT(setLinearInterp()));
    connect(ui->logInterpRadioButton, SIGNAL(clicked()), this, SLOT(setLogInterp()));

    connect(ui->debugModeCheckBox, SIGNAL(toggled(bool)), ui->debugNextPushButton, SLOT(setEnabled(bool)));
    connect(ui->debugModeCheckBox, SIGNAL(toggled(bool)), ui->debugAbortPushButton, SLOT(setEnabled(bool)));

    MainWindow *mainWinParent = static_cast<MainWindow*>(parent);
    connect(ui->debugNextPushButton, SIGNAL(clicked()), mainWinParent, SLOT(userDebugNext()));
}

OutputDialog::~OutputDialog()
{
    delete ui;
    if(m_listModel != NULL)
        delete m_listModel;
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
        dispErrMap();
        return;
    }

    if(m_data.size() == 0)
    {
        qDebug() << "ERROR: Setting level slice with no solution data";
        dispErrMap();
        return;
    }

    if(rects.size() == 0)
    {
        qDebug() << "ERROR: There are no rectangles in the drawing pipeline!";
        qDebug() << "Did you call the mesher?";
        dispErrMap();
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
            dispErrMap();
            return;
        }

        for(int ix = 0; ix < m_mesh->xMesh; ix++)
            for(int iy = 0; iy < m_mesh->yMesh; iy++)
            {
                float val = m_data[ix*m_mesh->yMesh*m_mesh->zMesh + iy*m_mesh->zMesh + level];
                if(val < minval)
                {
                    if(!m_logInterp || val > 0)  // Don't count 0 on log scale
                        minval = val;
                }
                if(val > maxval)
                    maxval = val;
            }

        if(maxval <= 1E-35)
        {
            qDebug() << "Zero flux everywhere!";
            dispErrMap();
            return;
        }

        if(minval < 0)
        {
            qDebug() << "WARNING: Negative flux!";
        }

        if((maxval - minval) / maxval < 1E-5)
        {
            qDebug() << "Displaying a flat surface!";
            dispErrMap();
            return;
        }

        qDebug() << "minval = " << minval << "  maxval = " << maxval;

        if(m_logInterp)
        {
            minval = log10(minval);
            maxval = log10(maxval);
            qDebug() << "log(minval) = " << minval << "  log(maxval) = " << maxval;
        }

        QStringList list;
        QString datastring = "";
        QString fidstring = "";
        for(int i = 0; i < m_mesh->xMesh; i++)
        {
            for(int j = 0; j < m_mesh->yMesh; j++)
            {
                float flux = m_data[i*m_mesh->yMesh*m_mesh->zMesh + j*m_mesh->zMesh + level];

                if(m_logInterp)
                    if(flux <= 1E-35)
                        rects[i*m_mesh->yMesh + j]->setBrush(errBrush);
                    else
                        flux = log10(flux);
                int fid = round(63*(flux-minval) / (maxval-minval));

                // TODO - Not sure how this can happen, but it seems to....
                if(fid > 63)
                {
                    qDebug() << "WARNING: fid > 63!";
                    qDebug() << "flux = " << flux << "  maxval = " << maxval;
                    fid = 63;
                }
                rects[i*m_mesh->yMesh + j]->setBrush(brushes[fid]);
                datastring += "   " + QString::number(flux);
                fidstring += "   " + QString::number(fid);
            }
            datastring += "\n";
            fidstring += "\n";
        }
        list << datastring;
        list << fidstring;

        // Makes the OutputDialog update the listview
        //m_listModel->setStringList(list);
    }
    else if(ui->xzRadioButton->isChecked())
    {
        if(level >= m_mesh->yMesh)
        {
            qDebug() << "level is too high! y-slices = " << m_mesh->yMesh;
            dispErrMap();
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

void OutputDialog::dispErrMap()
{
    for(unsigned int i = 0; i < rects.size(); i++)
        rects[i]->setBrush(errBrush);
}

void OutputDialog::setLinearInterp()
{
    m_logInterp = false;
    setSliceLevel(ui->sliceVerticalSlider->value());
}

void OutputDialog::setLogInterp()
{
    m_logInterp = true;
    setSliceLevel(ui->sliceVerticalSlider->value());
}

bool OutputDialog::debuggingEnabled()
{
    return ui->debugModeCheckBox->isChecked();
}





















