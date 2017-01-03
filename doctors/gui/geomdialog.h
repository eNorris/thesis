#ifndef GEOMDIALOG_H
#define GEOMDIALOG_H

#include <QDialog>
#include <QGraphicsScene>
#include <QGraphicsRectItem>

#include "gui/colormappable.h"

class Mesh;

namespace Ui {
class GeomDialog;
}

class GeomDialog : public QDialog, public ColorMappable
{
    Q_OBJECT

public:
    explicit GeomDialog(QWidget *parent = 0);
    ~GeomDialog();

    QGraphicsScene *scene;
    std::vector<QGraphicsRectItem*> rects;

    void updateMesh(Mesh *mesh);

private:
    Ui::GeomDialog *ui;
    Mesh *m_mesh;
    int m_rendertype;

protected slots:
    void setSliceLevel(int level);
    void updateMeshSlicePlane();
    void updateRenderType();
};

#endif // GEOMDIALOG_H
