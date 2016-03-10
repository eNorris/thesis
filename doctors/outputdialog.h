#ifndef OUTPUTDIALOG_H
#define OUTPUTDIALOG_H

#include <vector>

#include <QDialog>
#include <QGraphicsScene>
#include <QGraphicsRectItem>

namespace Ui {
class OutputDialog;
}

class OutputDialog : public QDialog
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
    std::vector<QBrush> brushes;

    std::vector<float> m_data;

    void loadParula();

protected slots:
    void disp(std::vector<float>);
};

#endif // OUTPUTDIALOG_H