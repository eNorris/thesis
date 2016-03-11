#ifndef QUADDIALOG_H
#define QUADDIALOG_H

#include <QDialog>

namespace Ui {
class QuadDialog;
}

class QuadDialog : public QDialog
{
    Q_OBJECT

public:
    explicit QuadDialog(QWidget *parent = 0);
    ~QuadDialog();

private:
    Ui::QuadDialog *ui;
};

#endif // QUADDIALOG_H
