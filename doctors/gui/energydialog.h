#ifndef ENERGYDIALOG_H
#define ENERGYDIALOG_H

#include <QDialog>
#include <QGraphicsScene>
#include <QGraphicsRectItem>
#include <vector>

class AmpxParser;

namespace Ui {
class EnergyDialog;
}

class EnergyDialog : public QDialog
{
    Q_OBJECT

public:
    explicit EnergyDialog(QWidget *parent = 0);
    ~EnergyDialog();

protected:
    bool m_xlog;
    bool m_ylog;
    std::vector<float> m_energyBins;

    QGraphicsScene *m_scene;
    std::vector<QGraphicsRectItem*> m_rects;
    std::vector<QGraphicsLineItem*> m_lines;

private:
    Ui::EnergyDialog *ui;

public slots:
    void setEnergy(AmpxParser *p);
};

#endif // ENERGYDIALOG_H
