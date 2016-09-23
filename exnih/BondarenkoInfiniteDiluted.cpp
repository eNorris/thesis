#include <stdio.h>
#include <string.h>
//#include "Standard/Interface/AbstractStream.h"
#include "BondarenkoInfiniteDiluted.h"
//#include "Resource/AmpxLib/ampxlib_config.h"
BondarenkoInfiniteDiluted::BondarenkoInfiniteDiluted()
{
    initialize();
}
BondarenkoInfiniteDiluted::BondarenkoInfiniteDiluted(const BondarenkoInfiniteDiluted& orig){
    d = orig.d;
    d->ref.ref();
}
void BondarenkoInfiniteDiluted::initialize()
{
    d = new BondarenkoInfiniteDilutedData();
}
BondarenkoInfiniteDiluted::~BondarenkoInfiniteDiluted()
{
    if( !d->ref.deref() ){
        delete d;
    }
}
bool BondarenkoInfiniteDiluted::operator==(BondarenkoInfiniteDiluted & a){
    if( d == a.d ) return true;
    if( d->size != a.d->size ) return false;
    for( int i = 0; i < d->size; i++ ){
        if( d->values[i] != a.d->values[i] ) return false;
    }
    return true;
}
BondarenkoInfiniteDiluted * BondarenkoInfiniteDiluted::getCopy() const
{
    BondarenkoInfiniteDiluted * copy = new BondarenkoInfiniteDiluted(*this);
    return copy;
}

// Serialization interfaces

const long BondarenkoInfiniteDiluted::uid = 0x6baff95172442622;

/**
 * @brief Serialize the object into a contiguous block of data
 * @param AbstractStream * stream - the stream into which the contiguous data will be stored
 * @return int - 0 upon success, error otherwise
 */
/*
int BondarenkoInfiniteDiluted::serialize(Standard::AbstractStream * stream) const {
    if( stream == NULL ) return -1;    
    long classUID = this->getUID();
    stream->write((char*)&classUID, sizeof(classUID));
    
    // write the size of this object
    unsigned long serializedSize = getSerializedSize();
    stream->write((char*)&serializedSize, sizeof(serializedSize));
    
    // DBC=4 - capture the oldWriteHead for checksumming later
    Remember(long oldWriteHead = stream->getWriteHead());
    
    // write the data size  
    stream->write((char*)&d->size, sizeof(d->size));
    
    // write the BondarenkoInfiniteDiluted data
    if( d->size > 0 ){
        stream->write((char*)d->values, sizeof(float)*d->size);
    } 
    
    // DBC=4 - checksum the expected serialized size and the actual serialized size
    Ensure( static_cast<unsigned long>(stream->getWriteHead() - oldWriteHead) == BondarenkoInfiniteDiluted::getSerializedSize() );
    return 0;
}
*/

/**
 * @brief deserialize the object from the given AbstractStream 
 * @param AbstractStream * stream - the stream from which the object will be inflated
 * @return int - 0 upon success, error otherwise
 */
/*
int BondarenkoInfiniteDiluted::deserialize(Standard::AbstractStream * stream){
    long read_uid = stream->getNextUID();
    
    // make sure we know how to parse the object in the stream
    if( read_uid != this->getUID() ) return Serializable::TypeMismatch;
    
    // skip over uid
    stream->ignore(sizeof(read_uid));

    // read objects serialized size
    unsigned long serializedSize;
    stream->read((char*)&serializedSize, sizeof(serializedSize));  
    
    // need to make sure data is only owned by this object
    qAtomicDetach(d);
    
    stream->read((char*)&d->size, sizeof(d->size));
    if( d->size > 0 ){
        if( d->values == NULL ) d->values = new float[d->size];
        stream->read((char*)d->values, sizeof(float)*d->size);
    }
    return 0;
}
*/
/**
 * @brief Obtain the size in bytes of this object when serialized 
 * @return unsigned long - the size in bytes of the object when serialized
 */
unsigned long BondarenkoInfiniteDiluted::getSerializedSize() const {
    unsigned long size = 0;
       
    size = sizeof(d->size) + sizeof(float)* d->size;
    return size;
}  
/**
 * @brief Obtain the universal version identifier
 * the uid should be unique for all Serializable objects such that 
 * an object factory can retrieve prototypes of the desired Serializable
 * object and inflate the serial object into a manageable object
 * @return long - the uid of the object
 */
long BondarenkoInfiniteDiluted::getUID() const {
    return BondarenkoInfiniteDiluted::uid;
}
