/* 
 * @file:   AbstractDataNode.h
 * @author: Jordan P. Lefebvre, lefebvrejp@ornl.gov. Robert A. Lefebvre, lefebvrera@ornl.gov
 *
 * Created on November 13, 2012, 10:19 AM
 */

#ifndef ABSTRACTDATANODE_H
#define	ABSTRACTDATANODE_H
#include <string>
namespace ScaleData {

  /**
   * @class AbstractDataNode
   * @brief Abstract data node for dynamic type storage
   */
  class AbstractDataNode {
  public:
    /**
     * Constructor
     */
    AbstractDataNode();
    /**
     * Copy constructor
     * @param orig const AbstractDataNode& orig
     */
    AbstractDataNode(const AbstractDataNode& orig);
    /**
     * Destructor
     */
    virtual ~AbstractDataNode();
    /**
     * @brief getCopy returns a deep copy of the node
     * @return deep copy of the node
     */
    virtual AbstractDataNode* getCopy() const = 0;
    /**
     * @brief produce a string representing this object
     * @return 
     */
    virtual std::string toString()const=0;
  };
}//namespace ScaleData
#endif	/* ABSTRACTDATANODE_H */

