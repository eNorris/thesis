#include "zoomablepannablegraphicsview.h"

#include <QMouseEvent>
#include <QApplication>
#include <qmath.h>

ZoomablePannableGraphicsView::ZoomablePannableGraphicsView(QWidget *parent) :
    QGraphicsView(parent),
    m_zoomFactorBase(1.002f),
    m_modifiers(Qt::NoModifier)  // Qt::ControlModifier
{

}

void ZoomablePannableGraphicsView::zoom(float factor) {
  scale(factor, factor);
  centerOn(target_scene_pos);
  QPointF delta_viewport_pos = target_viewport_pos - QPointF(viewport()->width() / 2.0,
                                                             viewport()->height() / 2.0);
  QPointF viewport_center = mapFromScene(target_scene_pos) - delta_viewport_pos;
  centerOn(mapToScene(viewport_center.toPoint()));
}

void ZoomablePannableGraphicsView::setModifiers(Qt::KeyboardModifiers mods)
{
    m_modifiers = mods;
}

bool ZoomablePannableGraphicsView::eventFilter(QObject *object, QEvent *event) {
  if (event->type() == QEvent::MouseMove) {
    QMouseEvent* mouse_event = static_cast<QMouseEvent*>(event);
    QPointF delta = target_viewport_pos - mouse_event->pos();
    if (qAbs(delta.x()) > 5 || qAbs(delta.y()) > 5) {
      target_viewport_pos = mouse_event->pos();
      target_scene_pos = mapToScene(mouse_event->pos());
    }
  } else if (event->type() == QEvent::Wheel) {
    QWheelEvent* wheel_event = static_cast<QWheelEvent*>(event);
    if (QApplication::keyboardModifiers() == m_modifiers) {
      if (wheel_event->orientation() == Qt::Vertical) {
        double angle = wheel_event->angleDelta().y();
        double factor = qPow(m_zoomFactorBase, angle);
        zoom(factor);
        return true;
      }
    }
  }
  Q_UNUSED(object)
  return false;
}
