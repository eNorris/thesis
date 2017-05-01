#ifndef ENERGYDIALOG_H
#define ENERGYDIALOG_H

#include <QDialog>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsRectItem>
#include <vector>

class AmpxParser;

namespace Ui {
class EnergyDialog;
}

class EnergyDialog;

class EnergyGraphicsView : public QGraphicsView
{
    Q_OBJECT

public:
    explicit EnergyGraphicsView(QWidget *parent = 0);
    ~EnergyGraphicsView();

protected:
    bool m_drag;
    int m_button;
    EnergyDialog *m_parent;

public slots:
    void mousePressEvent(QMouseEvent *event);
    void mouseDoubleClickEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void resizeEvent(QResizeEvent *event);
};

class EnergyDialog : public QDialog
{
    Q_OBJECT

public:
    explicit EnergyDialog(QWidget *parent = 0);
    ~EnergyDialog();

    bool isXLog();
    std::vector<float> *getEnergy();
    std::vector<QGraphicsRectItem*> *getRects();
    std::vector<QGraphicsLineItem*> *getLines();
    std::vector<float> getUserIntensity();

protected:
    bool m_xlog;
    bool m_ylog;
    std::vector<float> m_energyBins;

    QGraphicsScene *m_scene;
    std::vector<QGraphicsRectItem*> m_rects;
    std::vector<QGraphicsLineItem*> m_lines;

private:
    Ui::EnergyDialog *ui;

signals:
    void notifyOkClicked();

public slots:
    void setEnergy(AmpxParser *p);
    void on_energyLogXCheckBox_toggled(bool s);
    void setMinEnergy(float e);
    void setMaxEnergy(float e);

    void on_energyOkPushButton_clicked();
    void on_energyPresetComboBox_activated(int);
};

#endif // ENERGYDIALOG_H
