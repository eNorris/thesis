#include "mainwindow.h"
#include <QApplication>

//#include <ctime>

int main(int argc, char *argv[])
{
    srand(12345);
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}
