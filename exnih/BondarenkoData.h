#ifndef BONDARENKODATA_H
#define BONDARENKODATA_H
#include "LibraryItem.h"
#include "BondarenkoInfiniteDiluted.h"
#include "BondarenkoFactors.h"
#include <QList>
#include <QString>
#include <QAtomicInt>
//#include "Standard/Interface/Serializable.h"

/// Indicates the number of integer options 
/// in a bondarenko set. MT, NF, NL,... etc.
#define AMPX_BONDARENKO_DATA_INTEGER_OPTIONS 6
/// indicates the index of the MT variable
#define AMPX_BONDARENKO_MT 0
/// indicates the index of the NF variable
#define AMPX_BONDARENKO_NF 1
/// indicates the index of the NL variable
#define AMPX_BONDARENKO_NL 2
/// indicates the index of the ORDER variable
#define AMPX_BONDARENKO_ORDER 3
/// indicates the index of the IOFF variable
#define AMPX_BONDARENKO_IOFF 4
/// indicates the index of the NZ variable
#define AMPX_BONDARENKO_NZ 5
class BondarenkoDataData{
public:
    /// contains bondarenko items
    /// mt, nf, nl, order, ioff, and nz
    int data[AMPX_BONDARENKO_DATA_INTEGER_OPTIONS]; 
    QAtomicInt ref;
    BondarenkoDataData():ref(1){
        for( int i = 0; i < AMPX_BONDARENKO_DATA_INTEGER_OPTIONS; i++)
            data[i] = 0;
    }
    BondarenkoDataData(const BondarenkoDataData & orig):ref(1){
        for( int i = 0; i < AMPX_BONDARENKO_DATA_INTEGER_OPTIONS; i++)
            data[i] = orig.data[i];
    }
};
class BondarenkoData : public LibraryItem //, public Standard::Serializable{
{
private:
    BondarenkoInfiniteDiluted * infiniteDiluted;
    BondarenkoFactors * factors;
    BondarenkoDataData * d;
       
    void initialize();
public:
    BondarenkoData();
    BondarenkoData(const BondarenkoData & orig);
    ~BondarenkoData();
    
    bool operator==(BondarenkoData & a);
    
    /// Obtain a copy of this bondarenkoData
    /// BondarenkoData * - The copy
    BondarenkoData * getCopy() const;

    BondarenkoInfiniteDiluted* getInfiniteDiluted(){return this->infiniteDiluted;}
    void setInfiniteDiluted(BondarenkoInfiniteDiluted * infDil){this->infiniteDiluted=infDil;}
    BondarenkoFactors* getFactors(){return this->factors;}
    void setFactors(BondarenkoFactors * factors){this->factors=factors;}
    
    /// Obtain a data member of this bondarenko object
    /// these data members are the MT, NF, NL, ORDER, IOFF, and NZ
    /// and have enumerated definitions of 
    /// AMPX_BONDARENKO_MT, _NF, _NL, etc.
    /// @param int var - The variable desired
    /// @return int - the integer value associated with the var passed in. 
    int getData(int var){
        if( var < 0 || var > AMPX_BONDARENKO_DATA_INTEGER_OPTIONS ) return 0;
        return d->data[var];
    }// end of getData
    
    /// set the value for the variable associated with var
    /// these data members are the MT, NF, NL, ORDER, IOFF, and NZ
    /// and have enumerated definitions of 
    /// AMPX_BONDARENKO_MT, _NF, _NL, etc.
    /// @param int var - The variable desired
    void setData(int var, int value){
        if( var < 0 || var > AMPX_BONDARENKO_DATA_INTEGER_OPTIONS ) return;
        qAtomicDetach(d);
        d->data[var] = value;
    } // end of set data

    int getMt(){return getData(AMPX_BONDARENKO_MT);}
    void setMt(int MT);
    int getNf(){return getData(AMPX_BONDARENKO_NF);}
    void setNf(int NF){setData(AMPX_BONDARENKO_NF,NF);}
    int getNl(){return getData(AMPX_BONDARENKO_NL);}
    void setNl(int NL){setData(AMPX_BONDARENKO_NL,NL);}
    int getOrder(){return getData(AMPX_BONDARENKO_ORDER);}
    void setOrder(int ORDER){setData(AMPX_BONDARENKO_ORDER,ORDER);}
    int getIoff(){return getData(AMPX_BONDARENKO_IOFF);}
    void setIoff(int IOFF){setData(AMPX_BONDARENKO_IOFF,IOFF);}
    int getNz(){return getData(AMPX_BONDARENKO_NZ);}
    void setNz(int NZ){setData(AMPX_BONDARENKO_NZ,NZ);}
    QString toQString(){
        QString result;
        
        result=QString("BondarenkoData mt=%1 nf=%2 nl=%3 order=%4 ioff=%5 nz=%6\n").arg(getMt())
                                                               .arg(getNf())
                                                               .arg(getNl())
                                                               .arg(getOrder())
                                                               .arg(getIoff())
                                                               .arg(getNz());
        return result;
    }

public:
    /**
     * @brief the universal version identifier for this class
     */
    static const long uid;     
}; 
#endif
