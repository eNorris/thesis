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
    gui/outputdialog.cpp \
    gui/geomdialog.cpp \
    gui/quaddialog.cpp \
    gui/xsectiondialog.cpp \
    gui/colormappable.cpp \
    xs_reader/ampxparser.cpp \
    xs_reader/ampxrecordparsers.cpp \
    xs_reader/nuclidedata.cpp \
    mainwindow.cpp \
    config.cpp \
    quadrature.cpp \
    mesh.cpp \
    xsection.cpp \
    zoomablepannablegraphicsview.cpp \
    quadglwidget.cpp \
    outwriter.cpp \
    ctdatamanager.cpp \
    solverparamsdialog.cpp \
    legendre.cpp \
    materialutils.cpp \
    solver.cpp \
    histogram.cpp \
    mcnpwriter.cpp \
    gui/energydialog.cpp \
    cuda_link.cu \
    cuda_kernels.cu \
    sourceparams.cpp \
    solverparams.cpp

SOURCES -= cuda_kernels.cu \
           cuda_link.cu

HEADERS  += mainwindow.h \
    gui/outputdialog.h \
    gui/geomdialog.h \
    gui/quaddialog.h \
    gui/xsectiondialog.h \
    gui/colormappable.h \
    xs_reader/ampxparser.h \
    xs_reader/ampxrecordparsers.h \
    xs_reader/nuclidedata.h \
    config.h \
    quadrature.h \
    mesh.h \
    xsection.h \
    zoomablepannablegraphicsview.h \
    quadglwidget.h \
    outwriter.h \
    ctdatamanager.h \
    solverparamsdialog.h \
    legendre.h \
    materialutils.h \
    solver.h \
    histogram.h \
    mcnpwriter.h \
    globals.h \
    gui/energydialog.h \
    cuda_link.h \
    cuda_kernels.h \
    sourceparams.h \
    solverparams.h


HEADERS -= cuda_link.h \
    cuda_kernels.h

FORMS    += mainwindow.ui \
    gui/outputdialog.ui \
    gui/geomdialog.ui \
    gui/quaddialog.ui \
    gui/xsectiondialog.ui \
    solverparamsdialog.ui \
    gui/energydialog.ui

#DISTFILES += \
#    testinput.cfg

# Copies testinput.cfg from the source directory to the build directory (where the executable lives) so that it can be found and parsed
#   properly.
#install_it.path = $$OUT_PWD
#install_it.files = testinput.cfg
#INSTALLS += install_it


# ===== Extra stuff for CUDA =====
# From: https://cudaspace.wordpress.com/2012/07/05/qt-creator-cuda-linux-review/
DESTDIR     = $$system(pwd)
OBJECTS_DIR = $$DESTDIR/Obj

# C++ flags
QMAKE_CXXFLAGS_RELEASE =-O3

# Cuda sources
CUDA_SOURCES += cuda_kernels.cu \
                cuda_link.cu

# Path to cuda toolkit install
CUDA_DIR      = /usr/local/cuda-7.0
#CUDA_DIR      = D:/CUDA/win/v8.0/sdk
CUDA_SDK =

# Path to header and libs files
INCLUDEPATH  += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64     # For a 64 bits Operating system
LIBS += -lcudart -lcuda

# GPU architecture
CUDA_ARCH     = sm_35                # Titan Z
#CUDA_ARCH = 52     # GTX 960

# Here are some NVCC flags I've always used by default.
NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

# nvcc error printout format ever so slightly different from gcc
# http://forums.nvidia.com/index.php?showtopic=171651
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
                2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

message(cuda.commands: $$cuda.commands)
message(cuda.depend_command: $$cuda.depend_command)

# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda

