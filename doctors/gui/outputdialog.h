#ifndef OUTPUTDIALOG_H
#define OUTPUTDIALOG_H

#include <vector>

#include <QDialog>
#include <QGraphicsScene>
#include <QGraphicsRectItem>

#include "gui/colormappable.h"
#include "globals.h"

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
    void dispErrMap();

    bool debuggingEnabled();

private:

    bool m_logInterp;
    Ui::OutputDialog *ui;
    QGraphicsScene *scene;
    QStringListModel *m_listModel;

    std::vector<QGraphicsRectItem*> rects;

    std::vector<RAY_T> *m_raytracerData;
    std::vector<SOL_T> *m_solverData;
    std::vector<float> m_renderData;
    Mesh *m_mesh;

    float m_minvalGlobal, m_maxvalGlobal;
    float m_minvalGlobalLog, m_maxvalGlobalLog;


protected slots:
    void setSliceLevel(int level);
    void updateMeshSlicePlane();
    void updateRaytracerData(std::vector<RAY_T>*);
    void updateSolverData(std::vector<SOL_T>*);
    void reRender();
    void setLinearInterp();
    void setLogInterp();
    void refresh();
    void updateSelectedEnergy(int g);

public slots:
    void setEnergyGroups(unsigned int groups);
};

#endif // OUTPUTDIALOG_H
