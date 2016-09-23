/*
 * @file:   EnergyBounds.h
 * @author: Jordan P. Lefebvre, lefebvrejp@ornl.gov
 *
 * Created on August 22, 2013, 7:37 AM
 */

#ifndef ScaleData_ENERGYBOUNDS_H
#define	ScaleData_ENERGYBOUNDS_H
#include <vector>

namespace ScaleData {

    /**
     * @class EnergyBounds
     * @brief Basic class which wraps energy bounds with default type=float
     */
    template<class T = float>
    class EnergyBounds {
    public:
        /**
         * Default constructor
         */
        EnergyBounds();
        /**
         * Copy constructor
         * @param orig <b>const EnergyBounds&</b>
         */
        EnergyBounds(const EnergyBounds& orig);
        /**
         * Destructor
         */
        virtual ~EnergyBounds();
        /**
         * Returns the number of energy groups
         * @return <b>size_t</b>
         */
        size_t getSize() const;
        /**
         * @brief Get pointer to the bounds.
         * @return <b>T*</b>
         */
        const T* getBounds() const;
        /**
         * @brief set the energy bounds
         * @param new_bounds <b>const std::vector&lt;T&gt;</b>
         */
        void setBounds(const std::vector<T> new_bounds);
    protected:
        std::vector<T> bounds;
    private:

    };//class EnergyBounds

}//namespace ScaleData

//
// Templated code include
//
#include "EnergyBounds.i.h"
#endif	/* ScaleData_ENERGYBOUNDS_H */

