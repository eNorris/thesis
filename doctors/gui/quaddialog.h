#ifndef QUADDIALOG_H
#define QUADDIALOG_H

#include <QDialog>

class Quadrature;

namespace Ui {
class QuadDialog;
}

class QuadDialog : public QDialog
{
    Q_OBJECT

public:
    explicit QuadDialog(QWidget *parent = 0);
    ~QuadDialog();

    void updateQuad(Quadrature *quad);

private:
    Ui::QuadDialog *ui;

    Quadrature *m_quad;
};

#endif // QUADDIALOG_H
