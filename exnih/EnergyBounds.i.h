/*
 * @file:   EnergyBounds.cpp
 * @author: Jordan P. Lefebvre, lefebvrejp@ornl.gov
 *
 * Created on August 22, 2013, 7:37 AM
 */
#include <algorithm>

#include "EnergyBounds.h"
namespace ScaleData {

    template<class T>
    EnergyBounds<T>::EnergyBounds() {
    }

    template<class T>
    EnergyBounds<T>::EnergyBounds(const EnergyBounds& orig) : bounds(orig.bounds) {
    }

    template<class T>
    EnergyBounds<T>::~EnergyBounds() {
    }

    template<class T>
    size_t EnergyBounds<T>::getSize() const {
        return bounds.size();
    }//getSize

    template<class T>
    const T* EnergyBounds<T>::getBounds() const {
        return &(bounds[0]);
    }//getBounds;

    template<class T>
    void EnergyBounds<T>::setBounds(const std::vector<T> new_bounds) {
        //
        // Resize my bounds according to new bounds
        //
        this->bounds.resize(new_bounds.size());
        //
        // Copy new bounds
        //
        std::copy(new_bounds.begin(), new_bounds.end(), bounds.begin());
    }//setBounds
}//namespace ScaleData

