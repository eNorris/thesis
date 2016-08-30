#ifndef SOLVERPARAMSDIALOG_H
#define SOLVERPARAMSDIALOG_H

#include <QDialog>

namespace Ui {
class SolverParamsDialog;
}

class SolverParamsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SolverParamsDialog(QWidget *parent = 0);
    ~SolverParamsDialog();

private:
    Ui::SolverParamsDialog *ui;
};

#endif // SOLVERPARAMSDIALOG_H
