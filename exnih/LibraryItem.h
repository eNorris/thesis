#ifndef LIBRARY_ITEM_H
#define LIBRARY_ITEM_H
#include <fstream>
#include <iostream>
using namespace std;

/// This class references a library item
/// and assisted with keeping track of where
/// and if a library item (nuclideInfo, header, energybound, etc)
/// has been read.
class LibraryItem {
    void initialize();
public:
    LibraryItem();

    // virtual destructor.
    virtual ~LibraryItem() {}

    /// Obtain a copy of the libraryItem
    /// return LibraryItem * - The copy
    virtual LibraryItem * getCopy() const;
};
#endif
