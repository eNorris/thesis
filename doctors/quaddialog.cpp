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

    //if(m_quad != NULL)
    //    delete m_quad;
}

void QuadDialog::updateQuad(Quadrature *quad)
{
    m_quad = quad;
}
