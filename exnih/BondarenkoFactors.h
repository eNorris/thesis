#ifndef BONDARENKOFACTORS_H
#define BONDARENKOFACTORS_H
#include "LibraryItem.h"
#include "resources.h"
#include <QString>
#include <QAtomicInt>
//#include "Standard/Interface/Serializable.h"
class BondarenkoFactorsData{
public:
    QAtomicInt ref;
    int nSig0, nTemp, nGrps;
    float * values;

    BondarenkoFactorsData(): ref(1), nSig0(0),nTemp(0),nGrps(0),values(NULL){}
    BondarenkoFactorsData(const BondarenkoFactorsData & orig)
        :ref(1),nSig0(orig.nSig0), nTemp(orig.nTemp), nGrps(orig.nGrps),values(NULL){
        int size = nSig0* nTemp*nGrps;
        if( size == 0) return;
        values = new float[size];
        for( int i = 0; i < size; i ++)
            values[i] = orig.values[i];
    };
    virtual ~BondarenkoFactorsData(){
        if( values != NULL) delete [] values;
    }
};
class BondarenkoFactors: public LibraryItem //, public Standard::Serializable
{
private:
    BondarenkoFactorsData * d;
    void initialize();
public:
    BondarenkoFactors();
    BondarenkoFactors(const BondarenkoFactors & factors);
    ~BondarenkoFactors();
    
    bool operator==(BondarenkoFactors & a);

    /// Obtain a copy of the factors
    /// @return BondarenkoFactors * - the copy
    BondarenkoFactors * getCopy() const;

    float * getValues(){return d->values;}
    float getAt(int sig0, int temp, int grp){
        if( sig0 < 0 || sig0 >= d->nSig0 ) return 0.0;
        if( temp < 0 || temp >= d->nTemp ) return 0.0;
        if( grp < 0 || grp >= d->nGrps ) return 0.0;
        return d->values[(grp*d->nTemp*d->nSig0)+(temp*d->nSig0)+(sig0)];
    }
    void setAt(int sig0, int temp, int grp, float value){
        if( sig0 < 0 || sig0 >= d->nSig0 ) return ;
        if( temp < 0 || temp >= d->nTemp ) return ;
        if( grp < 0 || grp >= d->nGrps ) return ;
        qAtomicDetach(d);
        d->values[(grp*d->nTemp*d->nSig0)+(temp*d->nSig0)+(sig0)]=value;
    }
    int getSig0Size(){return d->nSig0;}
    int getTempSize(){return d->nTemp;}
    int getGrpSize(){return d->nGrps;}
    void setSizes(int numSig0, int numTemps, int numGrps);
    void setValues(float * values, int numSig0, int numTemps, int numGrps){
        qAtomicDetach(d);
        this->d->values = values;
        this->d->nGrps=numGrps;
        this->d->nTemp=numTemps;
        this->d->nSig0=numSig0;
    }
    
    QString toQString(){
        QString results = QString("Bondarenko Factors: numSig=%1 numTemp=%2 numGrps=%3\n")
                          .arg(d->nSig0).arg(d->nTemp).arg(d->nGrps);
        if( getValues() != NULL ){
            
            for( int s = 0, i=0; s <getSig0Size(); s++ ){
                for( int t = 0; t < getTempSize(); t++ ){ 
                    for( int g = 0; g < getGrpSize(); g++, i++ ){ 
                        results += QString("  factor[sig=%1][temp=%2][grp=%3]=%4\n").
                           arg(s).arg(t).arg(g).arg(getAt(s,t,g));
                    }
                }
            }
        }
        return results;
    }// end of toString   

    // serializable interface
    

public:
    /**
     * @brief the universal version identifier for this class
     */
    static const long uid;     
}; 
#endif
