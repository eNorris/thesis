#ifndef LIBRARYENERGYBOUNDS_H
#define LIBRARYENERGYBOUNDS_H
#include <iostream>
#include <fstream>
#include <vector>
//#include "Standard/Interface/Serializable.h"
#include "EnergyBounds.h"
#include "LibraryItem.h"
using namespace std;

class LibraryEnergyBounds : public ScaleData::EnergyBounds<float>,
public LibraryItem { //, public Standard::Serializable {
private:
    vector<float> lethargyBounds;
    void initialize();
public:
    LibraryEnergyBounds();
    ~LibraryEnergyBounds();

    bool operator==(LibraryEnergyBounds & a);

    LibraryEnergyBounds * getCopy() const;

    void resizeBounds(int size) {
        if (size <= 0) return;
        bounds.resize(size);
    }

    void resizeLethargy(int size) {
        if (size <= 0) return;
        lethargyBounds.resize(size);
    }

    float * getBounds() {
        return &(bounds[0]);
    }

    const float * getBounds() const {
        return &(bounds[0]);
    }

    void setBounds(const float * bounds, int size) {
        this->bounds.resize(size);
        for (int i = 0; i < size; i++)this->bounds[i] = bounds[i];
    }

    int getBoundsSize() const {
        return bounds.size();
    }

    float * getLethargyBounds() {
        return &(lethargyBounds[0]);
    }

    const float * getLethargyBounds() const {
        return &(lethargyBounds[0]);
    }

    void setLethargyBounds(const float * bounds, int size) {
        lethargyBounds.resize(size);
        for (int i = 0; i < size; i++)lethargyBounds[i] = bounds[i];
    }

    int getLethargySize() const {
        return lethargyBounds.size();
    }


public:
    /**
     * @brief the universal version identifier for this class
     */
    static const long uid;
};
#endif
