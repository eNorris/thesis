#ifndef PICTURELABEL_H
#define PICTURELABEL_H

#include <QLabel>

class PictureLabel : public QLabel
{
    Q_OBJECT

protected:
    QPixmap *m_pixmap;

public:
    PictureLabel(QWidget *parent);
    ~PictureLabel();

    void setPixmapFile(QString filename);

protected:
    //virtual void resizeEvent(int w, int h);
    //virtual void resizeEvent(const QSize& sz);
    virtual void resizeEvent(QResizeEvent *event);
};

#endif // PICTURELABEL_H

