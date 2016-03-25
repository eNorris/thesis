#include "quadglwidget.h"

#include <QMouseEvent>
#include <math.h>
#include <QDebug>

QuadGlWidget::QuadGlWidget(QWidget *parent) : QOpenGLWidget(parent),
    m_xRot(0),
    m_yRot(0),
    m_zRot(0)
{
    QSurfaceFormat format;
    format.setDepthBufferSize(24);
    setFormat(format);
}

QuadGlWidget::~QuadGlWidget()
{
    cleanup();
}

void QuadGlWidget::initializeGL()
{
    initializeOpenGLFunctions();
}

void QuadGlWidget::resizeGL(int w, int h)
{
    m_projection.setToIdentity();
    m_projection.perspective(60.0f, 2/float(h), 0.01f, 1000.0f);
}

void QuadGlWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // paint the picture
}

void QuadGlWidget::setXRotation(int angle)
{
    // Use float later so this isn't necessary
    while (angle < 0)
        angle += 360 * 16;
    while (angle > 360 * 16)
        angle -= 360 * 16;

    m_xRot = angle;
    update();
}

void QuadGlWidget::setYRotation(int angle)
{
    while (angle < 0)
        angle += 360 * 16;
    while (angle > 360 * 16)
        angle -= 360 * 16;

    m_yRot = angle;
    update();
}

void QuadGlWidget::setZRotation(int angle)
{
    while (angle < 0)
        angle += 360 * 16;
    while (angle > 360 * 16)
        angle -= 360 * 16;

    m_zRot = angle;
    update();
}

void QuadGlWidget::mousePressEvent(QMouseEvent *event)
{
    qDebug() << "QuadGlWidget::Press event";
    m_lastPos = event->pos();
}

void QuadGlWidget::mouseMoveEvent(QMouseEvent *event)
{
    qDebug() << "QuadGlWidget::Move event";
    int dx = event->x() - m_lastPos.x();
    int dy = event->y() - m_lastPos.y();

    if(event->buttons() & Qt::LeftButton)
    {
        setXRotation(m_xRot + 8 * dy);
        setYRotation(m_yRot + 8 * dx);
    }
    else if(event->buttons() & Qt::RightButton)
    {
        setXRotation(m_xRot + 8 * dy);
        setZRotation(m_zRot + 8 * dx);
    }
    m_lastPos = event->pos();
}

void QuadGlWidget::cleanup()
{
    makeCurrent();
    doneCurrent();
}
