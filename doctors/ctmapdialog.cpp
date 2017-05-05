#include "ctmapdialog.h"
#include "ui_ctmapdialog.h"

CtMapDialog::CtMapDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::CtMapDialog)
{
    ui->setupUi(this);
}

CtMapDialog::~CtMapDialog()
{
    delete ui;
}

void CtMapDialog::on_ctmapWaterPushButton_clicked()
{
    emit water();
    hide();
}

void CtMapDialog::on_ctmapPhantom19PushButton_clicked()
{
    emit phantom19();
    hide();
}
