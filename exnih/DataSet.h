/*
 * @file:   DataSet.h
 * @author: Jordan P. Lefebvre, lefebvrejp@ornl.gov. Robert A. Lefebvre, lefebvrera@ornl.gov
 *
 * Created on November 13, 2012, 10:19 AM
 */

#ifndef DATASET_H
#define DATASET_H
#include <string>
#include <map>
#include <vector>
#include <typeinfo>
#include "AbstractDataNode.h"

namespace ScaleData {

  /**
   * @class DataSet
   * @brief  data set for dynamic type storage and relations
   */
  class DataSet : public AbstractDataNode {
  private:    
    typedef std::map<std::string, AbstractDataNode*> DataMap;
    /**
     * std::map<string,AbstractDataSet*>
     */
    DataMap data;
  public:
    /**
     * Constructor
     */
    DataSet();
    /**
     * Copy constructor
     * @param orig const DataSet& orig
     */
    DataSet(const DataSet& orig);
    /**
     * Destructor
     */
    virtual ~DataSet();
    /**
     * Determine if this dataset is empty
     * @return true, if and only if the dataset contains data, false otherwise
     */
    bool empty()const{return data.empty();}
    /**
     * Get the value of this type
     */
    template<class T>
    const T& get(const std::string& key) const;
    template<class T>
    T& get(const std::string& key);
    /**
     * Get the reference to value of this type
     */
    template<class T>
    T& operator[](const std::string& key);
    /**
     * Get the value of this type with a default value (if there is no value in the dataset)
     */
    template<class T>
    T get(const std::string& key, T default_value) const;
    /**
     * Set the key,value
     * @param key std::string
     * @param value template<T>
     */
    template<class T>
    void put(const std::string& key, T value);
    /**
     * Set the key,value as a reference
     * @param key std::string
     * @param value template<T>
     */
    template<class T>
    void putRef(const std::string& key, T&value);
    /**
     * does this data set contain key value
     * @param key std::string
     * @return true if the data set contains it as is of type T, false otherwise
     */
    template<class T>
    bool contains(const std::string& key) const;

    /**
     * does this data set contain key
     * @param key std::string
     * @return true if the data set contains it, false otherwise
     */
    bool containsKey(const std::string& key) const;

    /**
     * Removes the given key,value
     * @param key std::string
     * @return T value removed
     */
    template<class T>
    T remove(const std::string& key);

    /**
     * Get the number of objects in this DataSet
     * @return size_t
     */
    size_t size() const;

    /**
     * Get keys for this dataset
     * @return std::vector<std::string> of keys
     */
    std::vector<std::string> keys() const;

    AbstractDataNode* getCopy() const;

    virtual std::string toString()const;
    DataSet& operator=(const DataSet& orig);
  };

}//namespace ScaleData
//
// Include inline and template memebers
//
#include "DataSet.i.h"
#endif  /* DATASET_H */

