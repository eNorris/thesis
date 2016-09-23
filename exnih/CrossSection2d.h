#ifndef CROSSSECTION2D_H
#define CROSSSECTION2D_H
#include "LibraryItem.h"
#include "ScatterMatrix.h"
#include <QList>
#include <QString>
#include <QtAlgorithms>
#include <QAtomicInt>
//#include "Standard/Interface/Serializable.h"

#define CROSS_SECTION_2D_INTEGER_OPTIONS 4
#define CROSS_SECTION_2D_MT 0
#define CROSS_SECTION_2D_LENGTH 1
#define CROSS_SECTION_2D_NL 2
#define CROSS_SECTION_2D_NT 3

class CrossSection2dData{
public :
    int data[CROSS_SECTION_2D_INTEGER_OPTIONS];
    QAtomicInt ref;
    int type;
    CrossSection2dData(): ref(1), type(0){
        for( int i = 0; i < CROSS_SECTION_2D_INTEGER_OPTIONS;i ++)
            data[i] = 0;
    };
    CrossSection2dData(const CrossSection2dData & orig): ref(1){
        for( int i = 0; i < CROSS_SECTION_2D_INTEGER_OPTIONS;i ++)
            data[i] = orig.data[i];
        type = orig.type;
    }
};

class CrossSection2d : public LibraryItem//, public Standard::Serializable
{
private:

    QList<ScatterMatrix *> scatterMatrices;
    CrossSection2dData * d;
    void initialize();

public:
    CrossSection2d();
    CrossSection2d(const CrossSection2d & orig);
    const CrossSection2d & operator=(const CrossSection2d & orig);
    ~CrossSection2d();

    bool operator==(CrossSection2d & a);

    /// Obtain a copy of the CrossSection2d
    /// @return CrossSection2d * - the copy
    CrossSection2d * getCopy() const;

    /// obtain the type of cross section data.
    ///  neutron2d, gamma2d, or gammaproduction
    /// AMPXLIB_NEUTRON2D_DATA,_GAMMA2D_DATA, or GAMMAPRODUCTION_DATA
    int getType(){return d->type;}

    /// Obtain a list of ascending temperature data
    /// for all ScatterMatrices associated with CrossSection2d
    QList<float> getTemperatureList();
    QList<ScatterMatrix *> * getScatterMatrices(){return &scatterMatrices;}
    int getNumScatterMatrices(){return scatterMatrices.size();}
    /// set the type of cross section
    /// @param int type - the type of data, one of the following:
    /// AMPXLIB_NEUTRON2D_DATA,_GAMMA2D_DATA, or GAMMAPRODUCTION_DATA
    void setType(int type){
        qAtomicDetach(d);
        this->d->type=type;
    }
    int * getData(){ return d->data;}
    int getData(int var){
        if( var < 0 || var >= CROSS_SECTION_2D_INTEGER_OPTIONS ) return 0;
        return d->data[var];
    }
    void setData(int var, int value){
        if( var < 0 || var >= CROSS_SECTION_2D_INTEGER_OPTIONS ) return ;
        qAtomicDetach(d);
        d->data[var] = value;
    }
    /// enumerated getters and setters
    int getMt(){return getData(CROSS_SECTION_2D_MT);}
    void setMt(int mt);
    /// Calculate and return the max length of any ScatterMatrix contained
    /// in this CrossSection2d
    int getLength();
    int getLegendreOrder(){return getData(CROSS_SECTION_2D_NL);}
    void setLegendreOrder(int legendreOrder){setData(CROSS_SECTION_2D_NL,legendreOrder);}
    int getNumberTemperatures(){return getData(CROSS_SECTION_2D_NT);}
    void setNumberTemperatures(int nt){setData(CROSS_SECTION_2D_NT,nt);}

    /// Add a ScatterMatrix to this CrossSection2d
    /// @param ScatterMatrix * scatterMatrix - the scatterMatrix to add
    /// @return bool - true, if successfully added
    ///                false, otherwise
    bool addScatterMatrix(ScatterMatrix * scatterMatrix){
        if( scatterMatrix == NULL ) return false;
        scatterMatrices.append(scatterMatrix);
        quickSort(scatterMatrices);
        if( getLegendreOrder() != scatterMatrices.back()->getLegendreOrder() ){
            setLegendreOrder(scatterMatrices.back()->getLegendreOrder());
        }

        return true;
    }
    /*!
     * Obtain a scatter matrix for the given temperature and legendre
     * @param int legendre - the desire legendre order for which to retrieve scatter matrix
     * @param float temperature - the desire temperature for which to retrieve scatter matrix
     * @return ScatterMatrix * - the scatter matrix with the given temperature and legendre order, NULL if not occurs
     */
    ScatterMatrix * getScatterMatrix(int legendreOrder, float temperature){
        QList<ScatterMatrix*> list = scatterMatrices;
        for( int i = 0; i < list.size(); i++){
            if( list[i]->getTemp() == temperature && list[i]->getLegendreOrder() == legendreOrder )
                return list.value(i);
        }
        return NULL;
    }
    /*!
     * Remove any and all scatter matrices with the given legendreOrder, and temperature
     * @param int legendreOrder - the legendre order of the desired scatter matrix
     * @param float temperature - the temperature of the desired scatter matrix
     * @return bool - true, if the matrix is removed, false other wise.
     */
    bool removeScatterMatrix(int legendreOrder, float temperature){
        ScatterMatrix * matrix = getScatterMatrix(legendreOrder, temperature);
        if( matrix == NULL ) return false;

        scatterMatrices.removeAll(matrix);
        if( scatterMatrices.size() == 0 ){
            if( getLegendreOrder() != 0 ) setLegendreOrder(0);
        }
        else if( getLegendreOrder() != scatterMatrices.back()->getLegendreOrder() ){
            setLegendreOrder(scatterMatrices.back()->getLegendreOrder());
        }
        delete matrix;
        return true;
    }
    /// remove the scattermatrix at the given index
    /// NOTE: The scatterMatrix at the given index will be deleted
    /// @param int index - the index at which to remove
    /// @return bool - true, if successfully added, false otherwise
    bool removeScatterMatrixAt(int index){
        if( index < 0 || index >= scatterMatrices.size() ) return false;
        ScatterMatrix * s = scatterMatrices.value(index);
        delete s;
        scatterMatrices.removeAt(index);
        return true;
    }

    /// Get the ScatterMatrix at the given index
    /// @param int index - The index of the desired ScatterMatrix
    /// @return ScatterMatrix * - The desired ScatterMatrix, or null if index is out of bounds
    ScatterMatrix * getScatterMatrix(int index){
        if( index < 0 || index >= scatterMatrices.size()) return NULL;
        return this->scatterMatrices.value(index);
    }
    void setScatterMatrices(QList<ScatterMatrix *> scatterMatrices){
        for( int i = 0; i < scatterMatrices.size(); i++){
            this->addScatterMatrix(scatterMatrices.value(i));
        }
    }
    /// Obtain a list of ScatterMatrix with a specific legendreOrder and
    /// optional temperature.
    /// @param int legendreOrder - The legendreOrder associated with the desired ScatterMatrices
    /// @param float * temp=NULL - The optional temperature. If a temperature is provide,
    ///                            the list should only return a single ScatterMatrix.
    /// @return QList<ScatterMatrix*>* - The list of desired ScatterMatrices,
    ///                                  NULL if no ScatterMatrix is associated
    ///                                  with the given legendreOrder and temp for this CrossSection2d
    QList<ScatterMatrix*> * getScatterMatrices(int legendreOrder, float * temp=NULL);

    QString toQString(){
        QString result = QString("CrossSection2d MT L NL NT\n");
        for(int i =0; i < CROSS_SECTION_2D_INTEGER_OPTIONS; i++)
        {
             result+= QString("%1 ").arg(getData(i));
        }
        return result;
    }

public:
    /**
     * @brief the universal version identifier for this class
     */
    static const long uid;
};
#endif
