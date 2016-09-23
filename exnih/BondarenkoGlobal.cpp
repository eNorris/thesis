//#include "Standard/Interface/AbstractStream.h"
//#include "Resource/AmpxLib/ampxlib_config.h"
#include "BondarenkoGlobal.h"
BondarenkoGlobal::BondarenkoGlobal()
{
    initialize();
}
void BondarenkoGlobal::initialize()
{
    d = new BondarenkoGlobalData();
}
BondarenkoGlobal::~BondarenkoGlobal()
{
    if( !d->ref.deref() ){
        delete d;
    }
}
BondarenkoGlobal::BondarenkoGlobal(const BondarenkoGlobal & orig){
    d = orig.d;
    d->ref.ref();    
}
bool BondarenkoGlobal::operator==(BondarenkoGlobal & a){
    if( d == a.d ) return true;
    if( d->elo != a.d->elo ) return false;
    if( d->ehi != a.d->ehi ) return false;
    if( d->sig0Size != a.d->sig0Size ) return false;
    if( d->tempSize != a.d->tempSize ) return false;
    for( int i = 0; i < d->sig0Size; i++){
        if( d->sig0[i] != a.d->sig0[i] ) return false;
    }
    for( int i = 0; i < d->tempSize; i++){
        if( d->temp[i] != a.d->temp[i] ) return false;
    }
    return true;
}
BondarenkoGlobal * BondarenkoGlobal::getCopy() const
{
    BondarenkoGlobal * copy = new BondarenkoGlobal(*this);
    return copy;
}

// Serialization interfaces

const long BondarenkoGlobal::uid = 0xdeaaf95200140511;


