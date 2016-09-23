#-------------------------------------------------
#
# Project created by QtCreator 2016-09-22T09:28:10
#
#-------------------------------------------------

QT       += core gui
CONFIG   += c++11

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = exnih
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    AmpxReader.cpp \
    AmpxLibrary.cpp \
    BondarenkoData.cpp \
    BondarenkoFactors.cpp \
    BondarenkoGlobal.cpp \
    BondarenkoInfiniteDiluted.cpp \
    CrossSection1d.cpp \
    CrossSection2d.cpp \
    LibraryEnergyBounds.cpp \
    LibraryNuclide.cpp \
    NuclideResonance.cpp \
    ScatterMatrix.cpp \
    resources.cpp \
    LibraryHeader.cpp \
    LibraryItem.cpp \
    NuclideFilter.cpp \
    SubGroupData.cpp \
    SinkGroup.cpp \
    EndianUtils.cpp \
    FileStream.cpp \
    AmpxDataHelper.cpp \
    DataSet.cpp \
    Filter.cpp \
    AbstractDataNode.cpp

HEADERS  += mainwindow.h \
    AmpxReader.h \
    AmpxLibrary.h \
    BondarenkoData.h \
    BondarenkoFactors.h \
    BondarenkoGlobal.h \
    BondarenkoInfiniteDiluted.h \
    CrossSection1d.h \
    CrossSection2d.h \
    LibraryEnergyBounds.h \
    LibraryNuclide.h \
    NuclideResonance.h \
    resources.h \
    ScatterMatrix.h \
    Resource.h \
    LibraryHeader.h \
    LibraryItem.h \
    NuclideFilter.h \
    SubGroupData.h \
    LibrarySourceDefs.h \
    SinkGroup.h \
    AbstractCrossSection1d.h \
    EndianUtils.h \
    FileStream.h

FORMS    += mainwindow.ui
