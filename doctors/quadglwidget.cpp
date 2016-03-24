#include "quadglwidget.h"

QuadGlWidget::QuadGlWidget(QWidget *parent) : QOpenGLWidget(parent)
{
    QSurfaceFormat format;
    format.setDepthBufferSize(24);
    setFormat(format);
}

QuadGlWidget::~QuadGlWidget()
{

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
