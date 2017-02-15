#include "energydialog.h"
#include "ui_energydialog.h"

#include <QDebug>
#include <QMouseEvent>

#include "xs_reader/ampxparser.h"
#include "gui/colormappable.h"

EnergyGraphicsView::EnergyGraphicsView(QWidget *parent) : m_drag(false)
{

}

EnergyGraphicsView::~EnergyGraphicsView()
{

}

void EnergyGraphicsView::mousePressEvent(QMouseEvent *event)
{
    qDebug() << "Got an event!" << event->pos() << ", " << event->localPos() << ", " << event->screenPos() << ", " << event->windowPos() << ", " << event->x();
    qDebug() << "Scene: " << mapToScene(event->pos());
    m_drag = true;
}

void EnergyGraphicsView::mouseReleaseEvent(QMouseEvent *event)
{
    m_drag = false;
}

void EnergyGraphicsView::mouseMoveEvent(QMouseEvent *event)
{
    if(m_drag)
    {
        QGraphicsItem *item = itemAt(event->pos());

        QGraphicsRectItem *rect = dynamic_cast<QGraphicsRectItem*>(item);  //->mapRectFromItem()
        if(rect == NULL)
            return;
        //float width = rect->rect().width();
        //float y = rect->y();
        QPointF p = mapToScene(QPoint(0, event->y()));
        rect->setY(p.y());
        qDebug() << "Width: " << event->y();
    }
}

EnergyDialog::EnergyDialog(QWidget *parent) :
    QDialog(parent),
    m_xlog(false),
    m_ylog(false),
    m_energyBins(),
    m_scene(NULL),
    m_rects(),
    m_lines(),
    ui(new Ui::EnergyDialog)
{
    ui->setupUi(this);

    m_scene = new QGraphicsScene(this);
    ui->graphicsView->setScene(m_scene);

    //ui->graphicsView->setDragMode(QGraphicsView::ScrollHandDrag);
    ui->graphicsView->setInteractive(true);
    ui->graphicsView->setMouseTracking(true);
}

EnergyDialog::~EnergyDialog()
{
    delete ui;
}

void EnergyDialog::setEnergy(AmpxParser *p)
{
    m_energyBins.resize(p->getGammaEnergyGroups() + 1);
    size_t eBins = p->getGammaEnergy().size();

    for(unsigned int i = 0; i < eBins; i++)
    {
        m_energyBins[i] = p->getGammaEnergy()[i];
    }

    QBrush greenBrush(Qt::green);
    QBrush blackBrush(Qt::black);
    QPen blackPen(blackBrush, 0);

    for(unsigned int i = 1; i < eBins; i++)
    {
        m_rects.push_back(m_scene->addRect(m_energyBins[i-1], 0, m_energyBins[i] - m_energyBins[i-1], 1, Qt::NoPen, greenBrush));
    }

    for(unsigned int i = 0; i < eBins; i++)
    {
        m_lines.push_back(m_scene->addLine(m_energyBins[i], 0, m_energyBins[i], 1, blackPen));
    }

    //QRectF r = m_scene->sceneRect();

    m_scene->setSceneRect(m_energyBins[eBins-1], 0, m_energyBins[0], 1);
    ui->graphicsView->fitInView(m_scene->sceneRect());
}

void EnergyDialog::on_energyLogXCheckBox_toggled(bool isChkd)
{
    //qDebug() << "Fire!" << isChkd;

    unsigned int eBins = m_energyBins.size();

    if(isChkd)
    {
        for(unsigned int i = 1; i < m_lines.size(); i++)
        {
            QRectF old = m_rects[i-1]->rect();
            float x = log10(m_energyBins[i-1]);
            float width = log10(m_energyBins[i]) - log10(m_energyBins[i-1]);
            m_rects[i-1]->setRect(x, old.y(), width, old.height());
        }
        for(unsigned int i = 0; i < m_lines.size(); i++)
        {
            QLineF old = m_lines[i]->line();
            float x = log10(m_energyBins[i]);
            m_lines[i]->setLine(x, old.y1(), x, old.y2());
        }
        float xmin = log10(m_energyBins[eBins-1]);
        float xmax = log10(m_energyBins[0]);
        qDebug() << "Scaling to " << xmin << ", " << xmax;
        m_scene->setSceneRect(xmin, 0, xmax-xmin, 1);
        ui->graphicsView->fitInView(m_scene->sceneRect());
    }
    else
    {
        for(unsigned int i = 1; i < m_rects.size(); i++)
        {
            QRectF old = m_rects[i-1]->rect();
            float x = m_energyBins[i-1];
            float width = m_energyBins[i] - m_energyBins[i-1];
            m_rects[i-1]->setRect(x, old.y(), width, old.height());
        }
        for(unsigned int i = 0; i < m_lines.size(); i++)
        {
            QLineF old = m_lines[i]->line();
            float x = m_energyBins[i];
            m_lines[i]->setLine(x, old.y1(), x, old.y2());
        }
        float xmin = m_energyBins[eBins-1];
        float xmax = m_energyBins[0];
        qDebug() << "Scaling to " << xmin << ", " << xmax;
        m_scene->setSceneRect(xmin, 0, xmax-xmin, 1);
        ui->graphicsView->fitInView(m_scene->sceneRect());
    }
}











