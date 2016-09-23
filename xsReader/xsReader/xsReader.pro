#-------------------------------------------------
#
# Project created by QtCreator 2015-10-14T11:28:03
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = xsReader
TEMPLATE = app


SOURCES += main.cpp\
    mainwindow.cpp \
    ampxparser.cpp \
    ampxrecordparsers.cpp \
    nuclidedata.cpp \
    treeitem.cpp \
    treemodel.cpp \
    outwriter.cpp \
    legendre.cpp

HEADERS  += mainwindow.h \
    ampxparser.h \
    ampxrecordparsers.h \
    nuclidedata.h \
    treeitem.h \
    treemodel.h \
    outwriter.h \
    legendre.h

FORMS    += mainwindow.ui
