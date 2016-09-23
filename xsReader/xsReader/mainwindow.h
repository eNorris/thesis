#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include "ampxparser.h"
#include "treemodel.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

protected:
    AmpxParser parser;
    TreeModel *model;
    //ifstream binaryFile;

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

protected slots:
    void openFile();
    void parseHeader();
    void parseData();
    //void openFile(QString filename);
    //void closeFile();
    //void parseHeader();
    //void parseData();
    void displayError(QString msg);
};

#endif // MAINWINDOW_H
