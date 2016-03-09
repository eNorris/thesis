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
    mesh.cpp

HEADERS  += mainwindow.h \
    config.h \
    quadrature.h \
    mesh.h

FORMS    += mainwindow.ui
