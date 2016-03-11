#include "geomdialog.h"
#include "ui_geomdialog.h"

GeomDialog::GeomDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::GeomDialog)
{
    ui->setupUi(this);
}

GeomDialog::~GeomDialog()
{
    delete ui;
}
