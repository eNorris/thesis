/* 
 * @file:   DataNode.i.h
 * @author: Jordan P. Lefebvre, lefebvrejp@ornl.gov. Robert A. Lefebvre, lefebvrera@ornl.gov
 *
 * Created on November 13, 2012, 10:19 AM
 */

#ifndef DATANODE_I_H
#define	DATANODE_I_H

#include <sstream>
#include <typeinfo>

namespace ScaleData {
  template<class T>
  DataNode<T>::DataNode(T& v)
  : value(v) {
  }

  template<class T>
  DataNode<T>::DataNode(const DataNode& orig)
  : value(orig.value) {
  }
  template<class T>
  DataNode<T>::~DataNode()
  {
  }

  template<class T>
  T& DataNode<T>::getRef() {
    return value;
  }
  template<class T>
  const T& DataNode<T>::get() const {
    return value;
  }

  template<class T>
  void DataNode<T>::set(T& v) {
    this->value = v;
  }
  template<class T>
  std::string DataNode<T>::toString() const{
      std::stringstream stream;
      stream<<"type("<<typeid(T).name()<<")";
      return stream.str();
  }
}//namespace ScaleData
#endif	/* DATANODE_I_H */

