#ifndef GEOMDIALOG_H
#define GEOMDIALOG_H

#include <QDialog>

class Mesh;

namespace Ui {
class GeomDialog;
}

class GeomDialog : public QDialog
{
    Q_OBJECT

public:
    explicit GeomDialog(QWidget *parent = 0);
    ~GeomDialog();

    void updateMesh(Mesh *mesh);

private:
    Ui::GeomDialog *ui;
    Mesh *m_mesh;

protected slots:
    void setSliceLevel(int level);
};

#endif // GEOMDIALOG_H
