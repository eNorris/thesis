#ifndef CROSSSECTION1D_H
#define CROSSSECTION1D_H
#include "LibraryItem.h"
#include "resources.h"
#include <QString>
#include <QList>
#include <QAtomicInt>
//#include "Standard/Interface/Serializable.h"
#include "AbstractCrossSection1d.h"

class CrossSection1dData{
public:
    QAtomicInt ref;
    float mt;
    int size;
    float * values;
    CrossSection1dData(): ref(1), mt(0), size(0), values(NULL){}
    CrossSection1dData(const CrossSection1dData & orig): ref(1),mt(orig.mt),size(0), values(NULL){
        if( orig.size <= 0 ) return;
        values = new float[orig.size];
        for( int i = 0; i < orig.size; i ++){
            values[i] = orig.values[i];
        }
        size = orig.size;
    }
    ~CrossSection1dData(){
        if( values != NULL) delete [] values;
    }
};
class CrossSection1d: public LibraryItem,
                      public Standard::AbstractCrossSection1d
{
private:
    CrossSection1dData * d;
    void initialize();
public:
    CrossSection1d();
    CrossSection1d(const CrossSection1d & orig);
    ~CrossSection1d();

    bool operator==(CrossSection1d & a);

    /// Obtain a copy of the CrossSection1d
    /// CrossSection1d * - the copy
    virtual CrossSection1d * getCopy() const;
    float * getValues(){return d->values;};
    int getSize(){return d->size;};
    void setValues(float * values, int size){
        qAtomicDetach(d);
        if( this->d->values != NULL) delete [] this->d->values;
        this->d->values = values;
        this->d->size=size;
    }
    int getMt(){return (int)d->mt;}
    void setMt(int mt);
    float getAt(int index){
        if( index < 0 || index >= getSize()) return 0.0;
        return getValues()[index];
    }
    void setAt(int index, float value){
        if( index < 0 || index >= getSize()) return ;
        qAtomicDetach(d);
        getValues()[index]=value;
    }
    void setSize(int size){
        if( size <= 0 ) return;
        qAtomicDetach(d);

        float * newValues = new float[size];
        for( int i = 0; i < size; i++){
            newValues[i] = 0; // zero out memory
        }
        setValues(newValues,size);
    }
    QString toQString(){
        QString results = QString("CrossSection1d mt=%1:\n").arg(getMt());
        if( getValues() != NULL ){
            int length = getSize();
            for( int i = 0; i < length; i++ ){
                results += QString("  xs[mt=%1][grp=%2]=%3\n").arg(d->mt).arg(i+1).arg(getValues()[i]);
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
