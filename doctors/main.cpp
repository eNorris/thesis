#include "mainwindow.h"
#include <QApplication>

#include "legendre.h"
#include <ctime>

void myMessageOutput(QtMsgType type, const QMessageLogContext &context, const QString &msg)
{
    QByteArray localMsg = msg.toLocal8Bit();
    switch (type) {
    case QtDebugMsg:
        fprintf(stderr, "Debug: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    // QtInfoMsg is Qt 5.5 and higher
    //case QtInfoMsg:
    //    fprintf(stderr, "Info: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
    //    break;
    case QtWarningMsg:
        fprintf(stderr, "Warning: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    case QtCriticalMsg:
        fprintf(stderr, "Critical: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    case QtFatalMsg:
        fprintf(stderr, "Fatal: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        abort();
    default:
        fprintf(stderr, "???: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
        break;
    }
}

int main(int argc, char *argv[])
{
    SphericalHarmonic s;
    AssocLegendre al;

    std::cout << al(0, 0, .5) << std::endl;

    int l = 15;

    std::clock_t start;
    for(int i = 0; i < 100000; i++)
        for(int m = 0; m <= l; m++)
            al(l, m, 0.75);
        //std::cout << al(l, m, 0.75) << std::endl;
    std::cout << "Time: " << (std::clock() - start)/(double)(CLOCKS_PER_SEC/1000.0) << " msec" << std::endl;


    srand(12345);
    qInstallMessageHandler(myMessageOutput);
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}
