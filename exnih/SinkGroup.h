#ifndef SINKGROUP_H
#define SINKGROUP_H
#include <iostream>
#include <QVector>
//#include "Standard/Interface/Serializable.h"
namespace Standard{
   class AbstractStream;
}

using namespace std;

class SinkGroup //: public Standard::Serializable{
{
private:
    QVector<float> data;
    int start, end, grp;
    void initialize();
public: 
    SinkGroup();
    SinkGroup(const SinkGroup& orig);
    const SinkGroup& operator=(const SinkGroup & orig);
    ~SinkGroup();
    
    /**
     * @brief equality operator
     * Determine if this SinkGroup and SinkGroup 'a' are equal
     * @return bool - true, if this and 'a' are equal, false otherwise
     */
    bool operator==(SinkGroup & a);
         
    /// Obtain a copy of the sinkgroup
    /// @return SinkGroup * - The copy
    virtual SinkGroup * getCopy() const;

    inline void setStart(int newStart){
        if(0==this->end){ this->end = newStart; }
        else if(newStart > this->end){ return; }
        else if( newStart == this->start ) return;
        // case 1, new start is greater than old start
        // we need to remove existing preceeding values
        for( ;this->start < newStart ; this->start++){
            if( data.size() > 0 )
                data.pop_front();
        }
        // case 2, new start is less than old start
        // we need to add values preceeding values
        for( ; this->start > newStart; this->start--){
            data.prepend(0.0);
        }
    }
    inline int getStart(){return this->start;}
    inline void setEnd(int newEnd){
        if(newEnd < this->start) return;
        if( this->end == newEnd ) return;
        int newSize = newEnd - this->start;
        data.resize(newSize); 
        this->end = newEnd;
//        for( ;this->end > newEnd; this->end--) data.pop_front();
//        for( ;this->end < newEnd; this->end++) data.append(0.0);
    }
    inline int getEnd(){return this->end;}
     
    void setGrp(int grp){
        this->grp=grp;
    }
    int getGrp(){return this->grp;}
  
    /// resize the sinkgroup to encompass the given index
    /// @param int index - the index to encompass by this sink group  
    bool resize(int index){
        int newStart=getStart(), newEnd=getEnd();
        if( index < 0 ) return false;
        if( index >= getStart() && index < getEnd() ) return true;
        
        if( index < getStart() ){ /// index is before start
            newStart = index;
            setStart(newStart);            
        }else if( getStart() == getEnd() ){
            newStart = index;
            newEnd = index+1;
            data.append(0.0);
            this->start = newStart;
            this->end = newEnd;
        }else{ /// index is after end
            newEnd = index+1;
            setEnd(newEnd);
        }   
        return true;
    }    
    float get(int index){
        if( index < getStart() || index >= getEnd() ) return 0.0;
        return data.value(index-getStart());
    }
    bool set(int index, float val){
        if( index < 0 ) return false;
        if( index < getStart() || index >= getEnd() ){
            if( !resize(index) )return false;
        }  
        data[index-getStart()]=val;
        return true;
    } 
    //! Convenience method to flip order
    //! Ampx SinkGroups are in descending order,
    //! rather than logical ascending order
    //! start,end, group, endValue....startValue
    //! More logical to be start,end, group, startValue...endValue
    //! The Data comes off AmpxLibrary end to start. 
    //! Call this method to flip to logical start to end
    void reverseData();
    static QList<SinkGroup*> expandMagicWordArray(float * data, int length,bool verbose=false);
    

public:
    /**
     * @brief the universal version identifier for this class
     */    
    static const long uid;
};
#endif
