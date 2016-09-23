#ifndef SCATTERMATRIX_H
#define SCATTERMATRIX_H
#include "LibraryItem.h"
#include "resources.h"
#include "SinkGroup.h"
#include <QString>
#include <QList>
//#include "Standard/Interface/Serializable.h"

class ScatterMatrix: public LibraryItem//, public Standard::Serializable
{
private:
    float temp;
    int legendreOrder;    
    QList<SinkGroup*> sinkGroups;
    void initialize();
public:
    ScatterMatrix();
    ScatterMatrix(const ScatterMatrix & orig);
    const ScatterMatrix & operator=(const ScatterMatrix & orig);
    ~ScatterMatrix();
    
    bool operator==(ScatterMatrix & a);
    
    /// Obtain a copy of the ScatterMatrix
    /// @return ScatterMatrix * - The copy
    ScatterMatrix * getCopy() const;
    QList<SinkGroup*> * getSinkGroups(){return & sinkGroups;}
    /// Obtain the first and last groups
    /// that contain sink data.
    /// @param int & first - the first group to populate
    /// @param int & last - the last group to populate
    /// @return bool - true, if first and last were succesfully found
    ///                  false, otherwise.
    bool getFirstLastGroup(int & first, int & last);
    
    /// Obtain a collapsed data set 
    /// where sink groups are represented in 
    /// magic word form
    /// @param int & length - variable to populate with the 
    ///                      length of the array being returned
    /// @return float * - the collapsed data in array form
    float * getCollapsedMatrix(int & length);
    
    /// Remove a sink group with the given grp
    /// All sink groups with a group of grp are removed and deleted
    /// @param int grp
    /// @return int - the number of sinkgroups removed
    int removeSinkGroup(int grp){ 
        int count=0;
        for( int i = 0 ; i < sinkGroups.size(); i++)
        {
            if( sinkGroups.at(i)->getGrp()==grp ){
                SinkGroup * sinkGroup = sinkGroups.value(i);
                delete sinkGroup;
                sinkGroups.removeAt(i);
                i--;
                count++;
            }
        }
        return count;
    } // end of removeSinkGroup

    /// Add a sink group to this scatter matrix
    /// If the scatter matrix already contains a sinkgroup 
    /// with the given group, the old sink group will be removed(deleted)
    /// @param SinkGroup * - The sink group to add to the matrix
    /// @return bool - indicates if the addition was succesful
    bool addSinkGroup(SinkGroup * sinkGroup){
        if ( sinkGroup == NULL ) return false;
        sinkGroups.append(sinkGroup);
        return true;
    } // end of addSinkGroup

    void setSinkGroups(QList<SinkGroup*> sinks){
        this->sinkGroups=sinks;
    }
    SinkGroup * getSinkGroup(int index){
        if( index < 0 || index >= sinkGroups.size() ) return NULL;
        return sinkGroups.value(index);
    }
    SinkGroup * getSinkGroupByGrp(int grp){
        for( int i = 0; i < sinkGroups.size(); i++){
            if( sinkGroups[i]->getGrp() == grp ) return sinkGroups[i];
        }
        return NULL;
    }
    float getTemp(){return temp;}
    void setTemp(float temp){
        this->temp=temp;
    }
    int getLegendreOrder(){return legendreOrder;}
    void setLegendreOrder(int legendreOrder){
        this->legendreOrder=legendreOrder;
    }
    QString toQString(){
        int firstGroup, lastGroup;
        this->getFirstLastGroup(firstGroup,lastGroup);
        QString results = QString("ScatterMatrix temp=%1 legendre=%2 first-grp=%3 last-grp=%4\n")
                          .arg(getTemp()).arg(getLegendreOrder()).arg(firstGroup).arg(lastGroup);
        
        return results;
    }// end of toQString   
    bool operator<(ScatterMatrix & that){
        bool result = false;
        if( this->getTemp() < that.getTemp()) result = true;
        else if(this->getTemp() == that.getTemp() ) result = this->getLegendreOrder() < that.getLegendreOrder();
        return result;
    }
    

public:
    /**
     * @brief the universal version identifier for this class
     */
    static const long uid;    
}; 

/*!
 * @brief Performs a quick sort on the given list
 * @param QList<ScatterMatrix*> & list - the list to sort
 */
void quickSort(QList<ScatterMatrix*> & list);
#endif
