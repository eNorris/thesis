#ifndef QUADGLWIDGET_H
#define QUADGLWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMatrix4x4>

class QuadGlWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
public:
    QuadGlWidget(QWidget *parent = NULL);
    ~QuadGlWidget();

protected:
    void initializeGL() Q_DECL_OVERRIDE;
    void resizeGL(int w, int h) Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;
    void mousePressEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    void mouseMoveEvent(QMouseEvent *event) Q_DECL_OVERRIDE;

public slots:
    void setXRotation(int angle);
    void setYRotation(int angle);
    void setZRotation(int angle);
    void cleanup();

private:
    QMatrix4x4 m_projection;
    int m_xRot;
    int m_yRot;
    int m_zRot;
    QPoint m_lastPos;
};

#endif // QUADGLWIDGET_H
