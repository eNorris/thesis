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
    //void updateSolution(std::vector<float> *data);
    void dispErrMap();

    bool debuggingEnabled();

private:

    //static const unsigned int COL = 0;
    //static const unsigned int UNC = 1;
    //static const unsigned int TOT = 2;

    bool m_logInterp;
    Ui::OutputDialog *ui;
    QGraphicsScene *scene;
    QStringListModel *m_listModel;

    std::vector<QGraphicsRectItem*> rects;

    //std::vector<float> *m_data;
    std::vector<RAY_T> *m_raytracerData;
    std::vector<SOL_T> *m_solverData;
    std::vector<float> m_renderData;
    Mesh *m_mesh;

    float m_minvalGlobal, m_maxvalGlobal;
    float m_minvalGlobalLog, m_maxvalGlobalLog;

    //unsigned int m_renderState;


protected slots:
    void setSliceLevel(int level);
    void updateMeshSlicePlane();
    //void reRender(std::vector<float>*);
    void updateRaytracerData(std::vector<RAY_T>*);
    void updateSolverData(std::vector<SOL_T>*);
    void reRender();
    void setLinearInterp();
    void setLogInterp();
    void refresh();
    void setEnergy(int g);
};

#endif // OUTPUTDIALOG_H
