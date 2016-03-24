#include "quaddialog.h"
#include "ui_quaddialog.h"

#include "quadrature.h"

QuadDialog::QuadDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::QuadDialog),
    m_quad(NULL)
{
    ui->setupUi(this);
}

QuadDialog::~QuadDialog()
{
    delete ui;
}

void QuadDialog::updateQuad(Quadrature *quad)
{
    m_quad = quad;
}
