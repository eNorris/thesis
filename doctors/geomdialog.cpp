#include "geomdialog.h"
#include "ui_geomdialog.h"

#include "mesh.h"

GeomDialog::GeomDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::GeomDialog),
    m_mesh(NULL)
{
    ui->setupUi(this);
}

GeomDialog::~GeomDialog()
{
    delete ui;
}

void GeomDialog::updateMesh(Mesh *mesh)
{
    m_mesh = mesh;
}
