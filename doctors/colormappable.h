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

protected:
    enum ColorId {NONE, PARULA, UNIQUE};
    ColorId m_colorId;
};

#endif // COLORMAPPABLE_H
