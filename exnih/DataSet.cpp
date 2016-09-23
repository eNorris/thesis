/*
 * File:   DataSet.cpp
 * Author: raq
 *
 * Created on November 21, 2012, 12:49 PM
 */
#include <sstream>

#include "DataSet.h"

namespace ScaleData{
  DataSet::DataSet() : AbstractDataNode() {
  }

  DataSet::DataSet(const DataSet& orig) : AbstractDataNode(orig){

      for( DataMap::const_iterator i = orig.data.begin(), ie = orig.data.end(); i != ie; i++ ) {          
          data.insert(std::make_pair(i->first,i->second->getCopy()));
      }
  }

  DataSet::~DataSet() {
      DataMap::const_iterator iter;
      //
      // Loop over map deleting infrastructure nodes
      //
      for (iter = data.begin(); iter != data.end(); iter++) {
        delete iter->second;
      }
//      data.clear();
  }

  size_t DataSet::size() const {
    return data.size();
  }

  std::vector<std::string> DataSet::keys() const {
    std::vector<std::string> key_list(data.size());
    DataMap::const_iterator iter;
    //
    // Loop over map compiling list of keys
    //
    int i = 0;
    for (iter = data.begin(); iter != data.end(); iter++) {
      key_list[i++] = iter->first;
    }
    return key_list;
  }//keys

  bool DataSet::containsKey(const std::string& key) const {
    DataMap::const_iterator iter = data.find(key);
    if (iter == data.end()) return false;
    return true;
  }//containsKey

  AbstractDataNode* DataSet::getCopy() const {
      return new DataSet( *this );
  }

  std::string DataSet::toString() const{
      std::stringstream stream;
      stream <<"DataSet: "<<std::endl;
      DataMap::const_iterator iter;
      //
      // Loop over map
      //
      for (iter = data.begin(); iter != data.end(); iter++) {
          stream<<"'"<<iter->first<<"':"<<iter->second->toString()<<std::endl;
      }
      return stream.str();
  }
  DataSet& DataSet::operator =(const DataSet& orig)
  {
      if (this != &orig)
      { // self-assignment check expected

          for( DataMap::const_iterator i = orig.data.begin(), ie = orig.data.end();
               i != ie;
               i++ )
          {
              data[i->first] = i->second->getCopy();
          }
      }
      return *this;

  }//DataSet::operator=
}//namespace ScaleData
