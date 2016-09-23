/* 
 * @file:   DataNode.h
 * @author: Jordan P. Lefebvre, lefebvrejp@ornl.gov. Robert A. Lefebvre, lefebvrera@ornl.gov
 *
 * Created on November 13, 2012, 10:19 AM
 */

#ifndef DATANODE_H
#define	DATANODE_H
#include <string>

#include "AbstractDataNode.h"

namespace ScaleData {

  /**
   * @class DataNode
   * @brief  data node for dynamic type storage
   */
  template<class T>
  class DataNode : public AbstractDataNode {
  private:
    T value;
  public:
    /**
     * Constructor
     */
    DataNode(T& v);
    /**
     * Copy constructor
     * @param orig const DataNode& orig
     */
    DataNode(const DataNode& orig);
    /**
     * Destructor
     */
    virtual ~DataNode();
    
    AbstractDataNode* getCopy() const
    {
        return new DataNode(*this);
    }

    /**
     * Get the value of this type
     */
    T& getRef();
    const T& get() const;
    /**
     * Set the value of this type
     * @param value template<T>
     */
    void set(T& value);
    virtual std::string toString()const;
  }; // end pf DataNode
}//namespace ScaleData
//
// Include inline and template memebers
//
#include "DataNode.i.h"
#endif	/* DATANODE_H */

