#include "energydialog.h"
#include "ui_energydialog.h"

#include <QDebug>

#include "xs_reader/ampxparser.h"
#include "gui/colormappable.h"

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
