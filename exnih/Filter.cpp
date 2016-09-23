/*
 * @file:   Filter.cpp
 * @author: Jordan P. Lefebvre, lefebvrejp@ornl.gov
 *
 * Created on September 10, 2013, 8:22 AM
 */
#include "Filter.h"

namespace ScaleData {

    Filter::Filter(Filter::Policy mpolicy) : policy(mpolicy) {
    }

    Filter::Filter(const Filter& filter) : data(filter.data), policy(filter.policy) {
    }

}//namespace ScaleData
