#ifndef OUTPUTDIALOG_H
#define OUTPUTDIALOG_H

#include <vector>

#include <QDialog>
#include <QGraphicsScene>
#include <QGraphicsRectItem>

#include "colormappable.h"

namespace Ui {
class OutputDialog;
}

class OutputDialog : public QDialog, public ColorMappable
{
    Q_OBJECT

public:
    explicit OutputDialog(QWidget *parent = 0);
    ~OutputDialog();

private:
    Ui::OutputDialog *ui;
    QGraphicsScene *scene;
    QGraphicsRectItem *rect;

    std::vector<QGraphicsRectItem*> rects;
    //std::vector<QBrush> brushes;

    std::vector<float> m_data;

    //void loadParulaBrush();
    //void loadUniqueBrush();

protected slots:
    void disp(std::vector<float>);
};

#endif // OUTPUTDIALOG_H
