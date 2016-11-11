#ifndef OUTPUTDIALOG_H
#define OUTPUTDIALOG_H

#include <vector>

#include <QDialog>
#include <QGraphicsScene>
#include <QGraphicsRectItem>

#include "colormappable.h"

class Mesh;

class QStringListModel;

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
    void updateSolution(std::vector<float> *data);
    void dispErrMap();

    bool debuggingEnabled();

private:
    bool m_logInterp;
    Ui::OutputDialog *ui;
    QGraphicsScene *scene;
    //QGraphicsRectItem *rect;
    QStringListModel *m_listModel;

    std::vector<QGraphicsRectItem*> rects;

    std::vector<float> *m_data;
    Mesh *m_mesh;

    float m_minvalGlobal, m_maxvalGlobal;
    float m_minvalGlobalLog, m_maxvalGlobalLog;


protected slots:
    //void disp(std::vector<float>);
    void setSliceLevel(int level);
    void updateMeshSlicePlane();
    void reRender(std::vector<float>*);
    void setLinearInterp();
    void setLogInterp();
    void refresh();
    void setEnergy(int g);
};

#endif // OUTPUTDIALOG_H
