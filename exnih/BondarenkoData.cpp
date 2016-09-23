//#include "Standard/Interface/AbstractStream.h"
#include "BondarenkoData.h"
#include "BondarenkoInfiniteDiluted.h"
#include "BondarenkoFactors.h"
//#include "Resource/AmpxLib/ampxlib_config.h"

BondarenkoData::BondarenkoData()
{
    initialize();
} // end
BondarenkoData::BondarenkoData(const BondarenkoData & orig){
    d = orig.d;
    d->ref.ref();
    setInfiniteDiluted(NULL);
    setFactors(NULL);    
    if( orig.infiniteDiluted != NULL )
        setInfiniteDiluted(orig.infiniteDiluted->getCopy());
    if( orig.factors != NULL )
        setFactors(orig.factors->getCopy());
}
void BondarenkoData::initialize()
{
   setInfiniteDiluted(NULL);
   setFactors(NULL);
   d = new BondarenkoDataData();
}// end of initialize
BondarenkoData::~BondarenkoData()
{
    if( getInfiniteDiluted() != NULL ) delete getInfiniteDiluted();
    if( getFactors() != NULL ) delete getFactors();
    if( !d->ref.deref() ){
        delete d;
    }
}
bool BondarenkoData::operator==(BondarenkoData & a){
    if( infiniteDiluted != NULL && a.infiniteDiluted != NULL ){
        bool equal = *infiniteDiluted == *a.infiniteDiluted;
        if( !equal ) return false;
    }else if( infiniteDiluted != a.infiniteDiluted ){
        return false;
    }
    if( factors != NULL && a.factors != NULL ){
        bool equal = *factors == *a.factors;
        if( !equal ) return false;
    }else if( factors != NULL && a.factors == NULL){
        return false;
    }else if( factors == NULL && a.factors != NULL){
        return false;
    }
    if( d == a.d ) return true;
    
    for( int i = 0; i < AMPX_BONDARENKO_DATA_INTEGER_OPTIONS; i++ ){
        if( getData(i) != a.getData(i) ){
            return false;
        }
    }
    return true;
}
BondarenkoData * BondarenkoData::getCopy() const
{
    BondarenkoData * copy = new BondarenkoData(*this);
    return copy;
}

void BondarenkoData::setMt(int MT){
    int oldMt = getMt();
    if( oldMt == MT ) return;            
    this->setData(AMPX_BONDARENKO_MT,MT);
}

// Serialization interfaces

const long BondarenkoData::uid = 0x1e777951b2042692;


