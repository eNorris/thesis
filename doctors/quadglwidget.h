#ifndef QUADGLWIDGET_H
#define QUADGLWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMatrix4x4>
//#include <qopenglfunctions.h>

class QuadGlWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
public:
    QuadGlWidget(QWidget *parent = NULL);
    ~QuadGlWidget();

protected:
    void initializeGL() Q_DECL_OVERRIDE;
    void resizeGL(int w, int h) Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;

private:
    QMatrix4x4 m_projection;
};

#endif // QUADGLWIDGET_H
