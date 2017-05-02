#include "energydialog.h"
#include "ui_energydialog.h"

#include <QDebug>
#include <QMouseEvent>
#include <QGraphicsRectItem>
#include <QFileDialog>
#include <QMessageBox>

#include "xs_reader/ampxparser.h"
#include "gui/colormappable.h"

EnergyGraphicsView::EnergyGraphicsView(QWidget *parent) : m_drag(false), m_button(0), m_parent(NULL)  //m_energyPointer(NULL)
{
    if(parent != NULL)
        m_parent = static_cast<EnergyDialog*>(parent);
}

EnergyGraphicsView::~EnergyGraphicsView()
{

}

void EnergyGraphicsView::mousePressEvent(QMouseEvent *event)
{
    m_drag = true;
    m_button = event->button();

    unsigned int indx = 0;
    std::vector<float> *energy = m_parent->getEnergy();
    std::vector<QGraphicsRectItem*> *rects = m_parent->getRects();

    if(energy->size() == 0 || rects->size() == 0)
        return;

    if(m_parent->isXLog())
    {
        while(mapToScene(event->pos()).x() < log10((*energy)[indx]) && indx < energy->size())
            indx++;
    }
    else
    {
        while(mapToScene(event->pos()).x() < (*energy)[indx] && indx < energy->size())
            indx++;
    }

    if(indx == 0 || indx > rects->size())
    {
        qDebug() << "Outside the indexable region!";
        return;
    }

    m_parent->setMinEnergy((*energy)[indx]);
    m_parent->setMaxEnergy((*energy)[indx-1]);
}

void EnergyGraphicsView::mouseDoubleClickEvent(QMouseEvent *event)
{

    qDebug() << "Double click";

    unsigned int indx = 0;
    std::vector<float> *energy = m_parent->getEnergy();
    std::vector<QGraphicsRectItem*> *rects = m_parent->getRects();

    if(energy->size() == 0 || rects->size() == 0)
        return;

    if(m_parent->isXLog())
    {
        while(mapToScene(event->pos()).x() < log10((*energy)[indx]) && indx < energy->size())
            indx++;
    }
    else
    {
        while(mapToScene(event->pos()).x() < (*energy)[indx] && indx < energy->size())
            indx++;
    }

    if(indx == 0 || indx > rects->size())
    {
        qDebug() << "Outside the indexable region!";
        return;
    }

    //qDebug() << "processed";

    QGraphicsRectItem *rect = (*rects)[indx-1];

    QRectF r = rect->rect();
    rect->setRect(r.x(), r.y(), r.width(), 1);
}

void EnergyGraphicsView::mouseReleaseEvent(QMouseEvent *event)
{
    m_drag = false;
}

void EnergyGraphicsView::mouseMoveEvent(QMouseEvent *event)
{

    if(m_parent == NULL)
        return;

    if(m_drag)
    {
        unsigned int indx = 0;
        std::vector<float> *energy = m_parent->getEnergy();
        std::vector<QGraphicsRectItem*> *rects = m_parent->getRects();

        if(energy->size() == 0 || rects->size() == 0)
            return;

        if(m_parent->isXLog())
        {
            while(mapToScene(event->pos()).x() < log10((*energy)[indx]) && indx < energy->size())
                indx++;
        }
        else
        {
            while(mapToScene(event->pos()).x() < (*energy)[indx] && indx < energy->size())
                indx++;
        }

        if(indx == 0 || indx > rects->size())
        {
            qDebug() << "Outside the indexable region!";
            return;
        }

        QGraphicsRectItem *rect = (*rects)[indx-1];

        QRectF r = rect->rect();
        QPointF p = mapToScene(event->pos());
        if(m_button == Qt::MouseButton::LeftButton)
            rect->setRect(r.x(), r.y(), r.width(), p.y());  //setY(p.y());
        else if(m_button == Qt::MouseButton::RightButton)
            rect->setRect(r.x(), r.y(), r.width(), 0);
    }
}

void EnergyGraphicsView::resizeEvent(QResizeEvent *event)
{
    fitInView(scene()->sceneRect());
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

    ui->graphicsView->setInteractive(true);
    ui->graphicsView->setMouseTracking(true);
    ui->graphicsView->scale(1, -1);
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
        m_rects.push_back(m_scene->addRect(m_energyBins[i-1], 0, m_energyBins[i] - m_energyBins[i-1], 0, Qt::NoPen, greenBrush));
    }

    for(unsigned int i = 0; i < eBins; i++)
    {
        m_lines.push_back(m_scene->addLine(m_energyBins[i], 0, m_energyBins[i], 1, blackPen));
    }

    m_scene->setSceneRect(m_energyBins[eBins-1], 0, m_energyBins[0], 1);
    ui->graphicsView->fitInView(m_scene->sceneRect());
}

bool EnergyDialog::isXLog()
{
    return ui->energyLogXCheckBox->isChecked();
}

std::vector<float> *EnergyDialog::getEnergy()
{
    return &m_energyBins;
}

std::vector<QGraphicsRectItem*> *EnergyDialog::getRects()
{
    return &m_rects;
}

std::vector<QGraphicsLineItem*> *EnergyDialog::getLines()
{
    return &m_lines;
}

std::vector<float> EnergyDialog::getUserIntensity()
{
    std::vector<float> v(m_rects.size());
    for(unsigned int i = 0; i < v.size(); i++)
        v[i] = m_rects[i]->rect().height();
    return v;
}

void EnergyDialog::on_energyLogXCheckBox_toggled(bool isChkd)
{
    unsigned int eBins = m_energyBins.size();
    if(eBins == 0)
        return;

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

void EnergyDialog::setMinEnergy(float eV)
{
    QString str;
    switch(ui->energyUnitComboBox->currentIndex())
    {
    case 0:  // MeV
        str = QString::number(eV / 1e6);
        break;
    case 1:  // keV
        str = QString::number(eV / 1e3);
        break;
    case 2:  // eV
        str = QString::number(eV);
        break;
    }
    ui->energyMinLabel->setText(str);
}

void EnergyDialog::setMaxEnergy(float eV)
{
    QString str;
    switch(ui->energyUnitComboBox->currentIndex())
    {
    case 0:  // MeV
        str = QString::number(eV / 1e6);
        break;
    case 1:  // keV
        str = QString::number(eV / 1e3);
        break;
    case 2:  // eV
        str = QString::number(eV);
        break;
    }
    ui->energyMaxLabel->setText(str);
}

void EnergyDialog::on_energyOkPushButton_clicked()
{

    hide();

    emit notifyOkClicked();
}

void EnergyDialog::on_energyPresetComboBox_activated(int indx)
{
    QRectF old;
    switch(indx)
    {
    case 0:
        return;
    case 1:
        for(unsigned int i = 0; i < m_rects.size(); i++)
        {
            old = m_rects[i]->rect();
            m_rects[i]->setRect(old.x(), old.y(), old.width(), 0);
        }

        //m_rects[1]->prepareGeometryChange();
        old = m_rects[m_rects.size()-2]->rect();
        m_rects[m_rects.size()-2]->setRect(old.x(), old.y(), old.width(), 1);
        break;
    default:
        qDebug() << "Preset #" << indx << " is not recognized";
    }

    update();
}

void EnergyDialog::on_energyOpenPushButton_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this, "Open Spectrum Formatted Data File", "/media/data/thesis/doctors/data/", "SPEC (*.spec);;All Files (*)");

    if(filename.isEmpty())
    {
        qDebug() << "Error, failed to load spectrum data file";
        return;
    }

    std::vector<float> values;
    std::ifstream fin(filename.toStdString().c_str());

    if(fin.good())
    {
        float f;
        while(fin >> f)
        {
            values.push_back(f);
        }
    }

    // Check to make sure the correct number of groups were loaded
    if(values.size() != m_energyBins.size()-1)
    {
        QString errmsg = QString("The XS Data file has ") + QString::number(m_energyBins.size()) + " energy groups, but " + filename + " contained " + QString::number(values.size()) + " data points";
        QMessageBox::warning(this, "Data Size Mismatch", errmsg, QMessageBox::Close);
        return;
    }

    float sum = 0.0f;
    for(unsigned int iv = 0; iv < values.size(); iv++)
        sum += values[iv];

    for(unsigned int iv = 0; iv < values.size(); iv++)
    {
        QRectF old = m_rects[iv]->rect();
        m_rects[iv]->setRect(old.x(), old.y(), old.width(), values[iv]/sum);
    }

    return;
}






