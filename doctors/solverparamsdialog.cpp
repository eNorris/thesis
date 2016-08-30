#include "solverparamsdialog.h"
#include "ui_solverparamsdialog.h"

SolverParamsDialog::SolverParamsDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::SolverParamsDialog)
{
    ui->setupUi(this);
}

SolverParamsDialog::~SolverParamsDialog()
{
    delete ui;
}
