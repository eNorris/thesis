#include "LibraryItem.h"
LibraryItem::LibraryItem()
{
    initialize();
} // end of constructor

void LibraryItem::initialize()
{
}// end of initiliaze

LibraryItem * LibraryItem::getCopy() const
{
    LibraryItem * copy = new LibraryItem();
    return copy;
}// end of getCopy
