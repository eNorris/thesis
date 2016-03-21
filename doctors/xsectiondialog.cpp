#include "xsectiondialog.h"
#include "ui_xsectiondialog.h"

#include "xsection.h"

XSectionDialog::XSectionDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::XSectionDialog),
    m_xs(NULL)
{
    ui->setupUi(this);
}

XSectionDialog::~XSectionDialog()
{
    delete ui;

    //if(m_xs != NULL)
    //    delete m_xs;
}

void XSectionDialog::updateXs(XSection *xs)
{
    m_xs = xs;
}
