#ifndef COLORMAPPABLE_H
#define COLORMAPPABLE_H

#include <vector>

#include <QBrush>

class ColorMappable
{
public:
    ColorMappable();

    std::vector<QBrush> brushes;

    void loadParulaBrush();
    void loadUniqueBrush();
};

#endif // COLORMAPPABLE_H
