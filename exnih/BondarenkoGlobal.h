#ifndef BONDARENKO_GLOBAL_H
#define BONDARENKO_GLOBAL_H
#include "LibraryItem.h"
#include "resources.h"
#include <QString>
#include <QAtomicInt>
//#include "Standard/Interface/Serializable.h"

class BondarenkoGlobalData{
public:
    QAtomicInt ref;
    float * sig0, * temp;
    float elo, ehi;
    int sig0Size, tempSize;
    BondarenkoGlobalData(): ref(1), sig0(NULL),temp(NULL),elo(0), ehi(0)
                        ,sig0Size(0),tempSize(0) {}
    BondarenkoGlobalData(const BondarenkoGlobalData & orig)
        :ref(1), sig0(NULL), temp(NULL), elo(0), ehi(0),sig0Size(0),tempSize(0)
    {
        elo = orig.elo;
        ehi = orig.ehi;
        if( orig.sig0Size > 0 ){
            sig0 = new float[orig.sig0Size];
            sig0Size = orig.sig0Size;
            for( int i = 0; i < orig.sig0Size; i ++)
                sig0[i] = orig.sig0[i];
        }
        if( orig.tempSize > 0 ){
            temp = new float[orig.tempSize];
            tempSize = orig.tempSize;
            for( int i = 0; i < orig.tempSize; i++)
                temp[i] = orig.temp[i];
        }
    }
    ~BondarenkoGlobalData(){
        if( sig0 != NULL ) delete [] sig0;
        if( temp != NULL ) delete [] temp;
    }
};

class BondarenkoGlobal : public LibraryItem
{
private:
    BondarenkoGlobalData * d;
    void initialize();
public:
    BondarenkoGlobal();
    BondarenkoGlobal(const BondarenkoGlobal & orig);
    ~BondarenkoGlobal();
    bool operator==(BondarenkoGlobal & a);
    /// Obtain a copy of this BondarenkoGlobal data
    /// @return BondarenkoGlobal * - The copy
    BondarenkoGlobal * getCopy() const;

    /// obtain the lower energy  for which factors can apply in the
    /// case where they do not span all energy groups
    float getElo(){ return d->elo;}
    void setElo(float elo){
        qAtomicDetach(d);
        this->d->elo = elo;
    }

    /// obtain the higher energy  for which factors can apply in the
    /// case where they do not span all energy groups
    float getEhi(){ return d->ehi;}
    void setEhi(float ehi){
        qAtomicDetach(d);
        this->d->ehi = ehi;
    }
    
    float * getSig0(){return d->sig0;}
    int getSig0Size(){return d->sig0Size;}
    void setSig0(float * sig0, int size){
        qAtomicDetach(d);
        this->d->sig0=sig0;d->sig0Size=size;
    }
    float * getTemps(){return d->temp;}
    int getTempsSize(){return d->tempSize;}
    void setTemps(float * temp, int size){
        qAtomicDetach(d);
        this->d->temp=temp;d->tempSize=size;
    }
    
    QString toQString(){
        QString results = QString("elo=%1 ehi=%2\n").arg(getElo()).arg(getEhi());
        if( getSig0() != NULL ){
            int length = getSig0Size();
            for( int i = 0; i < length; i++ ){
                results += QString("  sig0[%1]=%2\n").arg(i).arg(getSig0()[i]);
            }
        }
        if( getTemps() != NULL ){
            int length = getTempsSize();
            for( int i = 0; i < length; i++ ){
                results += QString("  temp[%1]=%2\n").arg(i).arg(getTemps()[i]);
            }
        }
        return results;
    }// end of toQString   


public:
    /**
     * @brief the universal version identifier for this class
     */
    static const long uid;    
}; 
#endif
