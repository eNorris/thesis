#include "BondarenkoFactors.h"
//#include "Resource/AmpxLib/ampxlib_config.h"
//#include "Standard/Interface/AbstractStream.h"

#include <QDebug>

BondarenkoFactors::BondarenkoFactors()
{
    initialize();
}
void BondarenkoFactors::initialize()
{
    d = new BondarenkoFactorsData();
}
BondarenkoFactors::BondarenkoFactors(const BondarenkoFactors & factors){
    d = factors.d;
    d->ref.ref();
}
BondarenkoFactors::~BondarenkoFactors()
{
    if( !d->ref.deref() ){
        delete d;
    }
}
bool BondarenkoFactors::operator==(BondarenkoFactors & a){
    if( d == a.d ) return true;
    if( d->nSig0 != a.d->nSig0 ) return false;
    if( d->nTemp != a.d->nTemp ) return false;
    if( d->nGrps != a.d->nGrps ) return false;
    for( int i = 0; i < getSig0Size(); i++){
        for( int t = 0; t < getTempSize(); t++){
            for( int g = 0; g < getGrpSize(); g++){
                if( getAt(i,t,g) != a.getAt(i,t,g) ) return false;
            }
        }
    }
    return true;
}
BondarenkoFactors * BondarenkoFactors::getCopy() const
{
    BondarenkoFactors * copy = new BondarenkoFactors(*this);    
    return copy;
}
void BondarenkoFactors::setSizes(int numSig0, int numTemps, int numGrps){
    int size = numSig0*numTemps*numGrps;
    //Require( size > 0);
    //Ensure( this->d != NULL );
    qDebug() << "Setting bondarenko sizes nSig0 "<<numSig0<<", numTemps "<<numTemps<<", numGrps "<<numGrps;
    qDebug() << "Existing bondarenko sizes nSig0 "<<d->nSig0<<", numTemps "<<d->nTemp<<", numGrps "<<d->nGrps;
    if( this->d->nGrps != numGrps ){
        qAtomicDetach(d);
    }else if( this->d->nTemp != numTemps ){
        qAtomicDetach(d);
    }else if( this->d->nSig0 != numSig0 ){
        qAtomicDetach(d);
    }else{
        // the sizes do not need to be changed because they are the same            
        return;
    }
    // need to delete old data
    if( d->values != NULL ) delete d->values;
    d->values = new float[numSig0*numTemps*numGrps];
    d->nSig0 = numSig0;
    d->nTemp = numTemps;
    d->nGrps = numGrps;
    for( int i = 0; i < size; i++) d->values[i]=0.0;
}
// Serialization interfaces

const long BondarenkoFactors::uid = 0x6babf951b2b426b2;


