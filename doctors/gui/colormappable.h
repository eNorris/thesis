#ifndef COLORMAPPABLE_H
#define COLORMAPPABLE_H

#include <vector>

#include <QBrush>

class ColorMappable
{
public:
    ColorMappable();

    std::vector<QBrush> brushes;
    QBrush errBrush;

    void loadParulaBrush();
    void loadUniqueBrush();
    void loadPhantom19Brush();

    void loadViridis256Brush();

protected:
    enum ColorId {NONE, PARULA, UNIQUE, PHANTOM19, VIRIDIS256};
    ColorId m_colorId;
};

#endif // COLORMAPPABLE_H
