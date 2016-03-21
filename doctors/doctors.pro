#-------------------------------------------------
#
# Project created by QtCreator 2016-03-08T17:19:52
#
#-------------------------------------------------

QT       += core gui
CONFIG += c++11

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = doctors
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    config.cpp \
    quadrature.cpp \
    mesh.cpp \
    xsection.cpp \
    solvers.cpp \
    outputdialog.cpp \
    geomdialog.cpp \
    quaddialog.cpp \
    xsectiondialog.cpp \
    colormappable.cpp \
    zoomablepannablegraphicsview.cpp

HEADERS  += mainwindow.h \
    config.h \
    quadrature.h \
    mesh.h \
    xsection.h \
    solvers.h \
    outputdialog.h \
    geomdialog.h \
    quaddialog.h \
    xsectiondialog.h \
    colormappable.h \
    zoomablepannablegraphicsview.h

FORMS    += mainwindow.ui \
    outputdialog.ui \
    geomdialog.ui \
    quaddialog.ui \
    xsectiondialog.ui

DISTFILES += \
    testinput.cfg

unix|win32: LIBS += -lconfig++
