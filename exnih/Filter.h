#ifndef CORE_FILTER_H
#define CORE_FILTER_H
/**
 * @Author : Robert A. Lefebvre
 * @Date   : 04-16-2013
 * Filter class to allow for object component filtering.
 */

#include <string>
#include <stdexcept>
#include "DataSet.h"

namespace ScaleData{
/**
 * Filter class provides basic constructs for allowing sets of data to be
 * kept or ignored.
 * For use of this class see tests/ExampleFilter.h.
 *
 * Typical use will include writing a subclass and providing domain specific
 * filter constructs as seen in the tests/ExampleFilter.h class
 */
class Filter{

public:

    /**
     * How do we handle objects
     */
    enum Policy{
        Keep,
        Ignore
    };
private:
    /**
     * The generic data container for storing filter descriptions
     */
    DataSet data;
    /**
     * Policy for dealing with unspecified components
     */
    Policy policy;

public:
    /**
     * Construct a filter object
     */
    Filter(Policy mpolicy=Keep);
    Filter(const Filter& filter);
    virtual ~Filter(){};

    /**
     * Determine if this filters a certain given named component
     * @param key - the key/name of the component
     * @param class T - the type of the component that the key maps to
     * @return bool - true, if and only if, the component with the given key and type T are present
     */
    template<class T>
    bool contains(const std::string& key) const;

    /**
     * Obtain the object describing what to filter for the given key
     * @param key - the name of the object describing what to filter
     * @param class T - the type of the object describing what to filter
     * @return T - the object describing what to filter
     * @Note - if the object T is not contained, a std::runtime_error is thrown
     */
    template<class T>
    T  get(const std::string& key) const;

    /**
     * Set the description data structure for the given key filter
     * @param key - the key/name of the filter
     * @param filterDescription - the data/ data structure describing what to filter
     */
    template<class T>
    void set(const std::string& key, T filterDescription);

    Policy getGlobalPolicy()const { return policy;}

}; // end of class Filter
} // end of namespace
#include "Filter.i.h"
#endif // end of include gaurd
