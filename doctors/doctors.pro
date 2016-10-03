#-------------------------------------------------
#
# Project created by QtCreator 2016-03-08T17:19:52
#
#-------------------------------------------------

# Print some messages during qmake about the Qt install being used in case there are multiple installs
message(===== qmake start =====)
message(Qt version: $$[QT_VERSION])
message(Qt is installed in $$[QT_INSTALL_PREFIX])
message(Qt resources can be found in the following locations:)
message(Documentation: $$[QT_INSTALL_DOCS])
message(Header files: $$[QT_INSTALL_HEADERS])
message(Libraries: $$[QT_INSTALL_LIBS])
message(Binary files (executables): $$[QT_INSTALL_BINS])
message(Plugins: $$[QT_INSTALL_PLUGINS])
message(Data files: $$[QT_INSTALL_DATA])
message(Translation files: $$[QT_INSTALL_TRANSLATIONS])
message(Settings: $$[QT_INSTALL_SETTINGS])
message(Examples: $$[QT_INSTALL_EXAMPLES])
message(Demonstrations: $$[QT_INSTALL_DEMOS])

message(The working directory is: $$PWD)
message(The destination directory is: $$DESTDIR)
message(The target directory is: $$DESTDIR_TARGET)
message(The project will be built in: $$OUT_PWD)


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
    zoomablepannablegraphicsview.cpp \
    quadglwidget.cpp \
    outwriter.cpp \
    raytracer.cpp \
    ctdatamanager.cpp \
    solverparams.cpp \
    solverparamsdialog.cpp \
    legendre.cpp

HEADERS  += mainwindow.h \
    config.h \
    quadrature.h \
    mesh.h \
    xsection.h \
    outputdialog.h \
    geomdialog.h \
    quaddialog.h \
    xsectiondialog.h \
    colormappable.h \
    zoomablepannablegraphicsview.h \
    quadglwidget.h \
    outwriter.h \
    ctdatamanager.h \
    solverparams.h \
    solverparamsdialog.h \
    legendre.h

FORMS    += mainwindow.ui \
    outputdialog.ui \
    geomdialog.ui \
    quaddialog.ui \
    xsectiondialog.ui \
    solverparamsdialog.ui \

DISTFILES += \
    testinput.cfg

# Copies testinput.cfg from the source directory to the build directory (where the executable lives) so that it can be found and parsed
#   properly.
install_it.path = $$OUT_PWD
install_it.files = testinput.cfg
INSTALLS += install_it

# This project relies on libconfig++ being installed. To install it, download libconfig-X.X.tar.gz from http://www.hyperrealm.com/libconfig/  unzip.
#   Follow the directions in the INSTALL file. Then run "sudo ldconfig" to update the LD path variables so it can be found.
#unix|win32: LIBS += -lconfig++
