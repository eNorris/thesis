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

    connect(ui->sliceVerticalSlider, SIGNAL(sliderMoved(int)), ui->sliceSpinBox, SLOT(setValue(int)));
    connect(ui->sliceSpinBox, SIGNAL(valueChanged(int)), ui->sliceVerticalSlider, SLOT(setValue(int)));

    connect(ui->sliceSpinBox, SIGNAL(valueChanged(int)), this, SLOT(setSliceLevel(int)));
}

GeomDialog::~GeomDialog()
{
    delete ui;
}

void GeomDialog::updateMesh(Mesh *mesh)
{
    m_mesh = mesh;
}

void GeomDialog::setSliceLevel(int level)
{
    qDebug() << "Set the slice to " << level;
}
