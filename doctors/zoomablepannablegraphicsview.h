#ifndef ZOOMABLEPANNABLEGRAPHICSVIEW_H
#define ZOOMABLEPANNABLEGRAPHICSVIEW_H

#include <QGraphicsView>

class ZoomablePannableGraphicsView : public QGraphicsView
{
public:
    ZoomablePannableGraphicsView(QWidget *parent = NULL);

    void zoom(float factor);
    void setModifiers(Qt::KeyboardModifiers mods);

private:
    float m_zoomFactorBase;
    Qt::KeyboardModifiers m_modifiers;
    QPointF target_scene_pos, target_viewport_pos;

    bool eventFilter(QObject* object, QEvent* event);

signals:

public slots:
};

#endif // ZOOMABLEPANNABLEGRAPHICSVIEW_H
