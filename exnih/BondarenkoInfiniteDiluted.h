#ifndef BONDARENKOINFINITEDILUTED_H
#define BONDARENKOINFINITEDILUTED_H

#include "LibraryItem.h"
#include "resources.h"
#include <QString>
#include <QAtomicInt>
//#include "Standard/Interface/Serializable.h"
class BondarenkoInfiniteDilutedData{
public:
    QAtomicInt ref;
    int size;
    float * values;
    BondarenkoInfiniteDilutedData():ref(1), size(0), values(NULL){}
    BondarenkoInfiniteDilutedData(const BondarenkoInfiniteDilutedData & orig)
            : ref(1),size(0), values(NULL){
        if( orig.size <= 0 ) return;
        this->size = orig.size;
        this->values = new float[this->size];
        for( int i = 0; i < this->size; i++)
            this->values[i] = orig.values[i];
    }
    ~BondarenkoInfiniteDilutedData(){
        if( this->values != NULL ) delete [] this->values;
    }
};

class BondarenkoInfiniteDiluted: public LibraryItem //, public Standard::Serializable
{
private:
    BondarenkoInfiniteDilutedData * d;
    void initialize();
public:
    BondarenkoInfiniteDiluted();
    BondarenkoInfiniteDiluted(const BondarenkoInfiniteDiluted & orig);
    ~BondarenkoInfiniteDiluted();
    
    bool operator==(BondarenkoInfiniteDiluted & a);
    
    float * getValues(){return d->values;};
    int getSize(){return d->size;};
    void setValues(float * values, int size){
        qAtomicDetach(d);
        this->d->values = values;
        this->d->size=size;
    }
    void setSize(int size){
        if( size <= 0 || size == d->size) return;
        qAtomicDetach(d);

        float * newValues = new float[size];
        for( int i = 0; i < size; i++){
            newValues[i] = 0; // zero out memory
        }
        setValues(newValues,size);
    }
    /// Obtain a copy of the infinite diluted
    /// @return BondarenkoInfiniteDiluted * - the copy
    BondarenkoInfiniteDiluted * getCopy() const;
   

    float getAt(int index){
        if( index < 0 || index >= getSize()) return 0.0;
        return getValues()[index];
    }
    void setAt(int index, float value){
        if( index < 0 || index >= getSize()) return ;
        qAtomicDetach(d);
        getValues()[index]=value;     
    }
    QString toQString(){
        QString results = QString("Infinite Diluted:\n");
        if( getValues() != NULL ){
            int length = getSize();
            for( int i = 0; i < length; i++ ){
                results += QString("  InfDil[grp=%1]=%2\n").arg(i+1).arg(getAt(i));
            }
        }
        return results;
    }// end of toQString   
    
    // serializable interface
    
    /**
     * @brief Serialize the object into a contiguous block of data
     * @param Standard::AbstractStream * stream - the stream into which the contiguous data will be stored
     * @return int - 0 upon success, error otherwise
     */
    //virtual int serialize(Standard::AbstractStream * stream) const;
    
    /**
     * @brief deserialize the object from the given Standard::AbstractStream 
     * @param Standard::AbstractStream * stream - the stream from which the object will be inflated
     * @return int - 0 upon success, error otherwise
     */
    //virtual int deserialize(Standard::AbstractStream * stream);
    /**
     * @brief Obtain the size in bytes of this object when serialized 
     * @return unsigned long - the size in bytes of the object when serialized
     */
    virtual unsigned long getSerializedSize() const;    
    /**
     * @brief Obtain the universal version identifier
     * the uid should be unique for all Standard::Serializable objects such that 
     * an object factory can retrieve prototypes of the desired Standard::Serializable
     * object and inflate the serial object into a manageable object
     * @return long - the uid of the object
     */
    virtual long getUID() const;     
    /**
     * @brief Determine if this object is a child of the given parent UID
     * This is intended to assist in object deserialization, when an object
     * may have been subclassed, the appropriate subclass prototype must be 
     * obtained from the object factory in order to correctly deserialize and
     * validate object inheritance.
     * i.e.
     * Object X contains a list of object Y. Object Y has child class object Z.
     * When deserializing object X, we expect Y, or a child of Y to be 
     * deserialized, any other objects would be an error.
     * @return bool - true, if this object is a child of a class with the given uid, false otherwise
     */
    //virtual bool childOf(long parentUID) const {return Standard::Serializable::getUID()==parentUID || Standard::Serializable::childOf(parentUID);}
public:
    /**
     * @brief the universal version identifier for this class
     */
    static const long uid;     
}; 
#endif
