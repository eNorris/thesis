#ifndef GEOMDIALOG_H
#define GEOMDIALOG_H

#include <QDialog>

namespace Ui {
class GeomDialog;
}

class GeomDialog : public QDialog
{
    Q_OBJECT

public:
    explicit GeomDialog(QWidget *parent = 0);
    ~GeomDialog();

private:
    Ui::GeomDialog *ui;
};

#endif // GEOMDIALOG_H
