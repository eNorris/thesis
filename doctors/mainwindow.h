#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

class OutputDialog;

//#include "config.h"
//#include "quadrature.h"
//#include "mesh.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

    OutputDialog *outputDialog;
};

#endif // MAINWINDOW_H
