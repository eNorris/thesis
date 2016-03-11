#include "xsectiondialog.h"
#include "ui_xsectiondialog.h"

#include "xsection.h"

XSectionDialog::XSectionDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::XSectionDialog)
{
    ui->setupUi(this);
}

XSectionDialog::~XSectionDialog()
{
    delete ui;
}

void XSectionDialog::updateXs(XSection *xs)
{
    m_xs = xs;
}
