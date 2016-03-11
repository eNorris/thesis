#include "quaddialog.h"
#include "ui_quaddialog.h"

QuadDialog::QuadDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::QuadDialog)
{
    ui->setupUi(this);
}

QuadDialog::~QuadDialog()
{
    delete ui;
}
