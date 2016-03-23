#ifndef OUTPUTDIALOG_H
#define OUTPUTDIALOG_H

#include <vector>

#include <QDialog>
#include <QGraphicsScene>
#include <QGraphicsRectItem>

#include "colormappable.h"

class Mesh;

namespace Ui {
class OutputDialog;
}

class OutputDialog : public QDialog, public ColorMappable
{
    Q_OBJECT

public:
    explicit OutputDialog(QWidget *parent = 0);
    ~OutputDialog();

    void updateMesh(Mesh *mesh);
    void updateSolution(std::vector<float> data);
    void dispErrMap();

private:
    bool m_logInterp;
    Ui::OutputDialog *ui;
    QGraphicsScene *scene;
    //QGraphicsRectItem *rect;

    std::vector<QGraphicsRectItem*> rects;

    std::vector<float> m_data;
    Mesh *m_mesh;


protected slots:
    //void disp(std::vector<float>);
    void setSliceLevel(int level);
    void updateMeshSlicePlane();
    void reRender(std::vector<float>);
    void setLinearInterp();
    void setLogInterp();
};

#endif // OUTPUTDIALOG_H
