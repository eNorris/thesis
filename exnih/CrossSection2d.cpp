#include "CrossSection2d.h"
//#include "Resource/AmpxLib/ampxlib_config.h"
#include <QDebug>
//#include "Standard/Interface/AbstractStream.h"


CrossSection2d::CrossSection2d()
{
    initialize();
}
CrossSection2d::CrossSection2d(const CrossSection2d & orig){
    d = orig.d;
    d->ref.ref();
    for( int i = 0; i < orig.scatterMatrices.size(); i++)
        this->scatterMatrices.append(orig.scatterMatrices[i]->getCopy());
}
void CrossSection2d::initialize()
{
    d = new CrossSection2dData();
}
bool CrossSection2d::operator==(CrossSection2d & a){
    if( scatterMatrices.size() != a.scatterMatrices.size() ) return false;
    for( int i = 0; i < scatterMatrices.size(); i++){
        bool equal = *scatterMatrices[i] == *a.scatterMatrices[i];
        if( !equal ) {
            qDebug() << "CrossSection2d ScatterMatrix not equal for mt="<<getMt()<<" at index "<<i;
            return false;
        }
    }
    if( d == a.d ) return true;
    for( int i = 0; i < CROSS_SECTION_2D_INTEGER_OPTIONS;i ++)
        if( d->data[i] != a.d->data[i] )
        {
            qDebug() << "CrossSection2d option not equal for mt="<<getMt()<<" at index "<<i;
            return false;
        }
    return true;
}
CrossSection2d::~CrossSection2d()
{
    if( !d->ref.deref() ){
        delete d;
    }
    for( int i = 0; i < scatterMatrices.size(); i++){
        delete scatterMatrices.value(i);
    }
    scatterMatrices.clear();
}
int CrossSection2d::getLength()
{
    int maxLength = -1;
    if( scatterMatrices.size() == 0 ){
        cerr<<"***Warning: CrossSection2d contains no ScatterMatrices!"<<endl;
        return 0;
    }
    for( int i = 0; i < scatterMatrices.size(); i++)
    {
        int length;
        /// delete the newly allocated collapsed matrix
        /// only care about the length of the collapsed data
        float * data = scatterMatrices.value(i)->getCollapsedMatrix(length);
        if( data == NULL ){
            cerr<<"***Error: ScatterMatrix returned NULL collapsed matrix!"<<endl;
            return -1;
        }
        delete[] data;
        if( length > maxLength ) maxLength = length;
    }
    return maxLength;
}
CrossSection2d * CrossSection2d::getCopy() const
{
    CrossSection2d * copy = new CrossSection2d(*this);
    return copy;
}
QList<float> CrossSection2d::getTemperatureList()
{
    QList<float> temperatures;
    for( int i = 0; i < scatterMatrices.size(); i++ )
    {
        ScatterMatrix * scatterMatrix = scatterMatrices.at(i);
        if( !temperatures.contains(scatterMatrix->getTemp()) )
        {
            temperatures.append(scatterMatrix->getTemp());
        }
    }
    qSort(temperatures);
    return temperatures;
}
QList<ScatterMatrix*>* CrossSection2d::getScatterMatrices(int legendreOrder, float * temp)
{
    QList<ScatterMatrix*> workingList;
    /// obtain list with same legendreOrder
    for( int i = 0; i < scatterMatrices.size(); i++)
    {
        /// if the legendreOrder is the desired, need to check temperature
        if( scatterMatrices.value(i)->getLegendreOrder() == legendreOrder )
        {
            /// check to see if the temperature was provided
            if( temp != NULL)
            {
                    /// check to determine if the temperature matches
                    if( *temp == scatterMatrices.value(i)->getTemp() )
                    {
                        workingList.append(scatterMatrices.value(i));
                        break; /// we assume, that only one scatter matrix should exist
                               /// for a given legendreOrder and temperature so break from loop
                    }
            }
            /// if temperature is null we want all temperatures associated with this legendreOrder
            else{
                workingList.append(scatterMatrices.value(i));
            }
        }
    }
    /// if none were found, return NULL
    if( workingList.isEmpty() ) return NULL;

    return new QList<ScatterMatrix*>(workingList);
} // end of getScatterMatrices

void CrossSection2d::setMt(int mt){
    int oldmt = getMt();
    if( oldmt == mt ) return;
    setData(CROSS_SECTION_2D_MT,mt);
}


