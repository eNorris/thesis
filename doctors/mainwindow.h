#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

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
};

#endif // MAINWINDOW_H
