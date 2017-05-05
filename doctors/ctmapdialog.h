#ifndef CTMAPDIALOG_H
#define CTMAPDIALOG_H

#include <QDialog>

namespace Ui {
class CtMapDialog;
}

class CtMapDialog : public QDialog
{
    Q_OBJECT

public:
    explicit CtMapDialog(QWidget *parent = 0);
    ~CtMapDialog();

private:
    Ui::CtMapDialog *ui;

protected slots:
    void on_ctmapWaterPushButton_clicked();
    void on_ctmapPhantom19PushButton_clicked();

signals:
    void phantom19();
    void water();
};

#endif // CTMAPDIALOG_H
