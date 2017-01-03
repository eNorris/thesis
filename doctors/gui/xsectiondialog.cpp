#include "xsectiondialog.h"
#include "ui_xsectiondialog.h"

#include "xsection.h"
#include "xs_reader/ampxparser.h"

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
}

void XSectionDialog::updateXs(XSection *xs)
{
    m_xs = xs;
}

void XSectionDialog::setXs(AmpxParser *p)
{

}
