#include "globals.h"

#include "outputdialog.h"
#include "ui_outputdialog.h"

#include <QDebug>
#include <QString>
#include <QStringListModel>

#include "mainwindow.h"
#include "mesh.h"

OutputDialog::OutputDialog(QWidget *parent) :
    QDialog(parent),
    m_logInterp(false),
    ui(new Ui::OutputDialog),
    scene(NULL),
    m_listModel(NULL),
    rects(),
    m_data(NULL),
    m_mesh(NULL),
    m_minvalGlobal(1.0E35f),
    m_maxvalGlobal(-1.0E35f),
    m_minvalGlobalLog(-1.0E35f),
    m_maxvalGlobalLog(1.0E35f)
{
    ui->setupUi(this);

    m_listModel = new QStringListModel(this);
    ui->listView->setModel(m_listModel);

    scene = new QGraphicsScene(this);
    ui->graphicsView->setScene(scene);

    ui->graphicsView->setDragMode(QGraphicsView::ScrollHandDrag);
    ui->graphicsView->setInteractive(true);
    ui->graphicsView->setMouseTracking(true);

    connect(ui->sliceVerticalSlider, SIGNAL(sliderMoved(int)), ui->sliceSpinBox, SLOT(setValue(int)));
    connect(ui->sliceSpinBox, SIGNAL(valueChanged(int)), ui->sliceVerticalSlider, SLOT(setValue(int)));

    connect(ui->sliceSpinBox, SIGNAL(valueChanged(int)), this, SLOT(setSliceLevel(int)));

    connect(ui->xyRadioButton, SIGNAL(clicked()), this, SLOT(updateMeshSlicePlane()));
    connect(ui->xzRadioButton, SIGNAL(clicked()), this, SLOT(updateMeshSlicePlane()));
    connect(ui->yzRadioButton, SIGNAL(clicked()), this, SLOT(updateMeshSlicePlane()));
    connect(ui->energyComboBox, SIGNAL(activated(int)), this, SLOT(setEnergy(int)));

    connect(ui->linearInterpRadioButton, SIGNAL(clicked()), this, SLOT(setLinearInterp()));
    connect(ui->logInterpRadioButton, SIGNAL(clicked()), this, SLOT(setLogInterp()));
    connect(ui->levelScaleCheckBox, SIGNAL(clicked()), this, SLOT(refresh()));

    connect(ui->debugModeCheckBox, SIGNAL(toggled(bool)), ui->debugNextPushButton, SLOT(setEnabled(bool)));
    connect(ui->debugModeCheckBox, SIGNAL(toggled(bool)), ui->debugAbortPushButton, SLOT(setEnabled(bool)));

    //MainWindow *mainWinParent = static_cast<MainWindow*>(parent);
    //connect(ui->debugNextPushButton, SIGNAL(clicked()), mainWinParent, SLOT(userDebugNext()));

    for(int i = 0; i < 19; i++)
        ui->energyComboBox->addItem(QString::number(i+1));
}

OutputDialog::~OutputDialog()
{
    delete ui;
    if(m_listModel != NULL)
        delete m_listModel;

    if(m_data != NULL)
        delete m_data;
}

void OutputDialog::updateMesh(Mesh *mesh)
{
    m_mesh = mesh;

    // I don't know the geometry anymore, so it doesn't make sense to draw anything
    //updateMeshSlicePlane();
}

void OutputDialog::updateSolution(std::vector<float> *data)
{
    m_data = data;

    m_minvalGlobal = 1.0E35f;
    m_maxvalGlobal = -1.0E35f;
    float minGtZero = 1.0E35f;

    for(unsigned int i = 0; i < m_data->size(); i++)
    {
        if((*m_data)[i] > m_maxvalGlobal)
            m_maxvalGlobal = (*m_data)[i];
        if((*m_data)[i] < m_minvalGlobal)
            m_minvalGlobal = (*m_data)[i];
        if((*m_data)[i] < minGtZero && (*m_data)[i] > 0)  // Don't allow zero in log scale
            minGtZero = (*m_data)[i];
    }
    m_minvalGlobalLog = log10(minGtZero);
    m_maxvalGlobalLog = log10(m_maxvalGlobal);
}


void OutputDialog::reRender(std::vector<float> *data)
{
    updateSolution(data);
    setSliceLevel(ui->sliceVerticalSlider->value());
}


void OutputDialog::setSliceLevel(int level)
{
    const int energyGroup = ui->energyComboBox->currentIndex();

    if(level < 0)
    {
        level = ui->sliceVerticalSlider->value();
    }

    if(m_mesh == NULL)
    {
        qDebug() << "ERROR: setSliceLevel on a NULL mesh pointer!";
        dispErrMap();
        return;
    }

    if(m_data == NULL)
    {
        qDebug() << "ERROR: setSliceLevel on a NULL data pointer!";
        dispErrMap();
        return;
    }

    if(m_data->size() == 0)
    {
        qDebug() << "ERROR: Setting level slice with no solution data";
        dispErrMap();
        return;
    }

    if(rects.size() == 0)
    {
        qDebug() << "ERROR: There are no rectangles in the drawing pipeline!";
        qDebug() << "Did you call the mesher?";
        updateMeshSlicePlane();
    }

    loadParulaBrush();

    float minvalLevel = 1.0E35f;
    float maxvalLevel = -1.0E35f;

    if(ui->xyRadioButton->isChecked())
    {
        if(level >= (signed) m_mesh->zElemCt)
        {
            qDebug() << "level is too high! z-slices = " << m_mesh->zElemCt;
            dispErrMap();
            return;
        }

        // Get the min/max for this slice level
        for(unsigned int ix = 0; ix < m_mesh->xElemCt; ix++)
            for(unsigned int iy = 0; iy < m_mesh->yElemCt; iy++)
            {
                float val = (*m_data)[energyGroup*m_mesh->voxelCount() + ix*m_mesh->yElemCt*m_mesh->zElemCt + iy*m_mesh->zElemCt + level];
                if(val < minvalLevel)
                {
                    if(!m_logInterp || val > 0)  // Don't count 0 on log scale
                        minvalLevel = val;
                }
                if(val > maxvalLevel)
                    maxvalLevel = val;
            }

        if(maxvalLevel <= 1E-35)
        {
            qDebug() << "Zero flux everywhere!";
            dispErrMap();
            return;
        }

        if(minvalLevel < 0)
        {
            qDebug() << "WARNING: Negative flux!";
        }

        if((maxvalLevel - minvalLevel) / maxvalLevel < 1E-5)
        {
            //qDebug() << "Displaying a flat surface!";
            dispErrMap();
            return;
        }

        //qDebug() << "minvalglobal = " << m_minvalGlobal << "   maxvalGlobal = " << m_maxvalGlobal << "   minvalLevel = " << minvalLevel << "   maxvalLevel = " << maxvalLevel;

        if(m_logInterp)
        {
            minvalLevel = log10(minvalLevel);
            maxvalLevel = log10(maxvalLevel);
            //qDebug() << "log(minvalglobal) = " << m_minvalGlobalLog << "   log(maxvalGlobal) = " << m_maxvalGlobalLog << "log(minvalLevel) = " << minvalLevel << "  log(maxvalLevel) = " << maxvalLevel;
        }

        QStringList list;
        QString datastring = "";
        QString fidstring = "";
        for(unsigned int i = 0; i < m_mesh->xElemCt; i++)
        {
            for(unsigned int j = 0; j < m_mesh->yElemCt; j++)
            {
                float flux = (*m_data)[energyGroup*m_mesh->voxelCount() + i*m_mesh->yElemCt*m_mesh->zElemCt + j*m_mesh->zElemCt + level];

                if(m_logInterp)
                {
                    if(flux <= 1E-35)
                    {
                        rects[i*m_mesh->yElemCt + j]->setBrush(errBrush);
                        continue;
                    }
                    else
                    {
                        flux = log10(flux);
                    }
                }


                // Map the flux value to color space
                int fid;
                if(ui->levelScaleCheckBox->isChecked())
                {
                    fid = round(63*(flux-minvalLevel) / (maxvalLevel-minvalLevel));
                }
                else
                {
                    if(m_logInterp)
                        fid = round(63*(flux-m_minvalGlobalLog) / (m_maxvalGlobalLog-m_minvalGlobalLog));
                    else
                        fid = round(63*(flux-m_minvalGlobal) / (m_maxvalGlobal-m_minvalGlobal));
                }

                if(fid > 63)
                {
                    qDebug() << "WARNING: fid > 63!";
                    qDebug() << "flux = " << flux << "  maxvalLevel = " << maxvalLevel << "  maxvalGlobal = " << m_maxvalGlobal << "  log(maxvalGlobal) = " << m_maxvalGlobalLog;
                    fid = -1;
                }
                if(fid < 0)
                {
                    qDebug() << "WARNING: fid < 0!";
                    qDebug() << "flux = " << flux << "  maxvalLevel = " << maxvalLevel << "  maxvalGlobal = " << m_maxvalGlobal << "  log(maxvalGlobal) = " << m_maxvalGlobalLog;
                    fid = -1;
                }
                if(fid == -1)
                    rects[i*m_mesh->yElemCt + j]->setBrush(errBrush);
                else
                    rects[i*m_mesh->yElemCt + j]->setBrush(brushes[fid]);
                datastring += "   " + QString::number(flux);
                fidstring += "   " + QString::number(fid);
            }
            datastring += "\n";
            fidstring += "\n";
        }
        list << datastring;
        list << fidstring;
    }
    else if(ui->xzRadioButton->isChecked())
    {
        if(level >= (signed) m_mesh->yElemCt)
        {
            qDebug() << "level is too high! y-slices = " << m_mesh->yElemCt;
            dispErrMap();
            return;
        }

        // Get the min/max for this slice level
        for(unsigned int ix = 0; ix < m_mesh->xElemCt; ix++)
            for(unsigned int iz = 0; iz < m_mesh->zElemCt; iz++)
            {
                float val = (*m_data)[energyGroup*m_mesh->voxelCount() + ix*m_mesh->xjmp() + iz + level];
                if(val < minvalLevel)
                {
                    if(!m_logInterp || val > 0)  // Don't count 0 on log scale
                        minvalLevel = val;
                }
                if(val > maxvalLevel)
                    maxvalLevel = val;
            }

        if(maxvalLevel <= 1E-35)
        {
            qDebug() << "Zero flux everywhere!";
            dispErrMap();
            return;
        }

        if(minvalLevel < 0)
        {
            qDebug() << "WARNING: Negative flux!";
        }

        if((maxvalLevel - minvalLevel) / maxvalLevel < 1E-5)
        {
            //qDebug() << "Displaying a flat surface!";
            dispErrMap();
            return;
        }

        //qDebug() << "minvalglobal = " << m_minvalGlobal << "   maxvalGlobal = " << m_maxvalGlobal << "   minvalLevel = " << minvalLevel << "   maxvalLevel = " << maxvalLevel;

        if(m_logInterp)
        {
            minvalLevel = log10(minvalLevel);
            maxvalLevel = log10(maxvalLevel);
            //qDebug() << "log(minvalglobal) = " << m_minvalGlobalLog << "   log(maxvalGlobal) = " << m_maxvalGlobalLog << "log(minvalLevel) = " << minvalLevel << "  log(maxvalLevel) = " << maxvalLevel;
        }

        QStringList list;
        QString datastring = "";
        QString fidstring = "";
        for(unsigned int i = 0; i < m_mesh->xElemCt; i++)
        {
            for(unsigned int j = 0; j < m_mesh->zElemCt; j++)
            {
                float flux = (*m_data)[energyGroup*m_mesh->voxelCount() + i*m_mesh->xjmp() + j + level];

                if(m_logInterp)
                {
                    if(flux <= 1E-35)
                    {
                        rects[i*m_mesh->yElemCt + j]->setBrush(errBrush);
                        continue;
                    }
                    else
                    {
                        flux = log10(flux);
                    }
                }


                // Map the flux value to color space
                int fid;
                if(ui->levelScaleCheckBox->isChecked())
                {
                    fid = round(63*(flux-minvalLevel) / (maxvalLevel-minvalLevel));
                }
                else
                {
                    if(m_logInterp)
                        fid = round(63*(flux-m_minvalGlobalLog) / (m_maxvalGlobalLog-m_minvalGlobalLog));
                    else
                        fid = round(63*(flux-m_minvalGlobal) / (m_maxvalGlobal-m_minvalGlobal));
                }

                if(fid > 63)
                {
                    qDebug() << "WARNING: fid > 63!";
                    qDebug() << "flux = " << flux << "  maxvalLevel = " << maxvalLevel << "  maxvalGlobal = " << m_maxvalGlobal << "  log(maxvalGlobal) = " << m_maxvalGlobalLog;
                    fid = -1;
                }
                if(fid < 0)
                {
                    qDebug() << "WARNING: fid < 0!";
                    qDebug() << "flux = " << flux << "  maxvalLevel = " << maxvalLevel << "  maxvalGlobal = " << m_maxvalGlobal << "  log(maxvalGlobal) = " << m_maxvalGlobalLog;
                    fid = -1;
                }
                if(fid == -1)
                    rects[i*m_mesh->yElemCt + j]->setBrush(errBrush);
                else
                    rects[i*m_mesh->yElemCt + j]->setBrush(brushes[fid]);
                datastring += "   " + QString::number(flux);
                fidstring += "   " + QString::number(fid);
            }
            datastring += "\n";
            fidstring += "\n";
        }
        list << datastring;
        list << fidstring;
        /////////////////////////////////////////
        ///////////////////////////////////////
        //if(level >= (signed) m_mesh->yElemCt)
        //{
        //    qDebug() << "level is too high! y-slices = " << m_mesh->yElemCt;
        //    dispErrMap();
        //    return;
        //}

        //for(unsigned int i = 0; i < m_mesh->xElemCt; i++)
        //    for(unsigned int j = 0; j < m_mesh->zElemCt; j++)
        //    {
        //        int zid = m_mesh->zoneId[i*m_mesh->yElemCt*m_mesh->zElemCt + level*m_mesh->zElemCt + j];
        //        rects[i*m_mesh->zElemCt + j]->setBrush(brushes[zid]);
        //    }
    }
    else if(ui->yzRadioButton->isChecked())
    {
        if(level >= (signed) m_mesh->xElemCt)
        {
            qDebug() << "level is too high! x-slices = " << m_mesh->xElemCt;
            dispErrMap();
            return;
        }

        // Get the min/max for this slice level
        for(unsigned int iy = 0; iy < m_mesh->yElemCt; iy++)
            for(unsigned int iz = 0; iz < m_mesh->zElemCt; iz++)
            {
                float val = (*m_data)[energyGroup*m_mesh->voxelCount() + iy*m_mesh->yElemCt + iz + level];
                if(val < minvalLevel)
                {
                    if(!m_logInterp || val > 0)  // Don't count 0 on log scale
                        minvalLevel = val;
                }
                if(val > maxvalLevel)
                    maxvalLevel = val;
            }

        if(maxvalLevel <= 1E-35)
        {
            qDebug() << "Zero flux everywhere!";
            dispErrMap();
            return;
        }

        if(minvalLevel < 0)
        {
            qDebug() << "WARNING: Negative flux!";
        }

        if((maxvalLevel - minvalLevel) / maxvalLevel < 1E-5)
        {
            //qDebug() << "Displaying a flat surface!";
            dispErrMap();
            return;
        }

        //qDebug() << "minvalglobal = " << m_minvalGlobal << "   maxvalGlobal = " << m_maxvalGlobal << "   minvalLevel = " << minvalLevel << "   maxvalLevel = " << maxvalLevel;

        if(m_logInterp)
        {
            minvalLevel = log10(minvalLevel);
            maxvalLevel = log10(maxvalLevel);
            //qDebug() << "log(minvalglobal) = " << m_minvalGlobalLog << "   log(maxvalGlobal) = " << m_maxvalGlobalLog << "log(minvalLevel) = " << minvalLevel << "  log(maxvalLevel) = " << maxvalLevel;
        }

        QStringList list;
        QString datastring = "";
        QString fidstring = "";
        for(unsigned int i = 0; i < m_mesh->yElemCt; i++)
        {
            for(unsigned int j = 0; j < m_mesh->zElemCt; j++)
            {
                float flux = (*m_data)[energyGroup*m_mesh->voxelCount() + i*m_mesh->yElemCt + j + level];

                if(m_logInterp)
                {
                    if(flux <= 1E-35)
                    {
                        rects[i*m_mesh->yElemCt + j]->setBrush(errBrush);
                        continue;
                    }
                    else
                    {
                        flux = log10(flux);
                    }
                }


                // Map the flux value to color space
                int fid;
                if(ui->levelScaleCheckBox->isChecked())
                {
                    fid = round(63*(flux-minvalLevel) / (maxvalLevel-minvalLevel));
                }
                else
                {
                    if(m_logInterp)
                        fid = round(63*(flux-m_minvalGlobalLog) / (m_maxvalGlobalLog-m_minvalGlobalLog));
                    else
                        fid = round(63*(flux-m_minvalGlobal) / (m_maxvalGlobal-m_minvalGlobal));
                }

                if(fid > 63)
                {
                    qDebug() << "WARNING: fid > 63!";
                    qDebug() << "flux = " << flux << "  maxvalLevel = " << maxvalLevel << "  maxvalGlobal = " << m_maxvalGlobal << "  log(maxvalGlobal) = " << m_maxvalGlobalLog;
                    fid = -1;
                }
                if(fid < 0)
                {
                    qDebug() << "WARNING: fid < 0!";
                    qDebug() << "flux = " << flux << "  maxvalLevel = " << maxvalLevel << "  maxvalGlobal = " << m_maxvalGlobal << "  log(maxvalGlobal) = " << m_maxvalGlobalLog;
                    fid = -1;
                }
                if(fid == -1)
                    rects[i*m_mesh->yElemCt + j]->setBrush(errBrush);
                else
                    rects[i*m_mesh->yElemCt + j]->setBrush(brushes[fid]);
                datastring += "   " + QString::number(flux);
                fidstring += "   " + QString::number(fid);
            }
            datastring += "\n";
            fidstring += "\n";
        }
        list << datastring;
        list << fidstring;
        ////////////////////////////////////////
        ///////////////////////////////////////
        //if(level >= (signed) m_mesh->xElemCt)
        //{
        //    qDebug() << "level is too high! x-slices = " << m_mesh->xElemCt;
        //    return;
        //}

        //for(unsigned int i = 0; i < m_mesh->yElemCt; i++)
        //    for(unsigned int j = 0; j < m_mesh->zElemCt; j++)
        //    {
        //        int zid = m_mesh->zoneId[level*m_mesh->yElemCt*m_mesh->zElemCt + i*m_mesh->zElemCt + j];
         //       rects[i*m_mesh->zElemCt + j]->setBrush(brushes[zid]);
        //    }
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

void OutputDialog::dispErrMap()
{
    for(unsigned int i = 0; i < rects.size(); i++)
        rects[i]->setBrush(errBrush);
}

void OutputDialog::setLinearInterp()
{
    m_logInterp = false;
    refresh();
}

void OutputDialog::setLogInterp()
{
    m_logInterp = true;
    refresh();
}

bool OutputDialog::debuggingEnabled()
{
    return ui->debugModeCheckBox->isChecked();
}

void OutputDialog::refresh()
{
    setSliceLevel(ui->sliceVerticalSlider->value());
}

void OutputDialog::setEnergy(int g)
{
    setSliceLevel(-1);
}

















