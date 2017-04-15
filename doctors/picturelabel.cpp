#include "picturelabel.h"

#include <QDebug>

PictureLabel::PictureLabel(QWidget *parent) : QLabel(parent), m_pixmap(NULL)
{

}

PictureLabel::~PictureLabel()
{
    if(m_pixmap != NULL)
        delete m_pixmap;
}

void PictureLabel::setPixmapFile(QString filename)
{
    m_pixmap = new QPixmap(filename);
    setScaledContents(true);
    setPixmap(m_pixmap->scaledToWidth(width(), Qt::SmoothTransformation));
}

/*
void PictureLabel::resizeEvent(const QSize& sz)
{
    qDebug() << "resize 1";
    QLabel::resize(sz);
}

void PictureLabel::resizeEvent(const int w, const int h)
{
    qDebug() << "resize 2";
    QLabel::resize(w, h);
}
*/

void PictureLabel::resizeEvent(QResizeEvent *event)
{
    if(m_pixmap == NULL)
        return;
    setPixmap(m_pixmap->scaledToWidth(width(), Qt::SmoothTransformation));

    QLabel::resizeEvent(event);
}
