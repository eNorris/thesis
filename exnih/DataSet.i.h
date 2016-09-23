/*
 * @file:   DataSet.i.h
 * @author: Jordan P. Lefebvre, lefebvrejp@ornl.gov. Robert A. Lefebvre, lefebvrera@ornl.gov
 *
 * Created on November 13, 2012, 10:19 AM
 */

#ifndef DATASET_I_H
#define DATASET_I_H

//#include "DataSet.h"
#include "DataNode.h"
#include <iostream>
#include <stdexcept>

namespace ScaleData {

  template<class T>
  const T& DataSet::get(const std::string& key) const {
    DataMap::const_iterator iter = data.find(key);
    if (iter == data.end()) throw std::runtime_error("***Error: DataSet::get("+key+") has no results!");
    DataNode<T>* dataNode = dynamic_cast<DataNode<T>*> (iter->second);
    return dataNode->get();
  }//get
  template<class T>
  T& DataSet::get(const std::string& key){
    DataMap::iterator iter = data.find(key);
    if (iter == data.end()) throw std::runtime_error("***Error: DataSet::get("+key+") has no results!");
    DataNode<T>* dataNode = dynamic_cast<DataNode<T>*> (iter->second);
    return dataNode->getRef();
  }//get
  template<class T>
  T& DataSet::operator[](const std::string& key) {
    DataMap::iterator iter = data.find(key);
    if (iter == data.end()) throw "No results!";
    DataNode<T>* dataNode = dynamic_cast<DataNode<T>*> (iter->second);
    return dataNode->getRef();
  }//get

  template<class T>
  T DataSet::get(const std::string& key, T default_value) const {
    if (contains<T>(key)) {
      return get<T>(key);
    } else {
      return default_value;
    }
  }//get

  template<class T>
  void DataSet::put(const std::string& key, T value) {
    //
    // Check if key,value already exists
    //
    DataMap::iterator iter = data.find(key);
    if (iter != data.end()) {
      //
      // Delete previous node
      //
      delete iter->second;
    }
    AbstractDataNode* dataNode = new DataNode<T> (value);
    data[key] = dataNode;
  }//put
 
  template<class T>
  bool DataSet::contains(const std::string& key) const {
    DataMap::const_iterator iter = data.find(key);
    if (iter == data.end()) return false;
    DataNode<T>* dataNode = dynamic_cast<DataNode<T>*> (iter->second);
    if (dataNode == NULL){
        return false;
    }
    return true;
  }//contains

  template<class T>
  T DataSet::remove(const std::string& key) {
    DataMap::const_iterator iter = data.find(key);
    if (iter == data.end()) throw "No results!";
    DataNode<T>* dataNode = dynamic_cast<DataNode<T>*> (iter->second);
    //
    // Remove key from map
    //
    T v = dataNode->get();
    delete iter->second;
    data.erase(key);
    return v;
  }//remove

}//namespace ScaleData
#endif  /* DATASET_I_H */

