#include "ScatterMatrix.h"
//#include "Resource/AmpxLib/ampxlib_config.h"
#include <QDebug>
#include <limits>
//#include "Standard/Interface/AbstractStream.h"

namespace {
//! Union to allow reinterpreting of int -> float without breaking strict aliasing
union FloatIntUnion {
    int   magic_word;
    float magic_float;
};
} // end anonymous namespace

ScatterMatrix::ScatterMatrix()
{
    initialize();
}
ScatterMatrix::ScatterMatrix(const ScatterMatrix & orig){
    legendreOrder = orig.legendreOrder;
    temp = orig.temp;
    for( int i = 0; i < orig.sinkGroups.size(); i++){
        this->sinkGroups.append(orig.sinkGroups[i]->getCopy());
    }
}
const ScatterMatrix& ScatterMatrix::operator=(const ScatterMatrix & orig){
    legendreOrder = orig.legendreOrder;
    temp = orig.temp;
    for( int i = 0; i < orig.sinkGroups.size(); i++){
        this->sinkGroups.append(orig.sinkGroups[i]->getCopy());
    }
    return * this;
}
bool ScatterMatrix::operator==(ScatterMatrix & a){
    if( sinkGroups.size() != a.sinkGroups.size() ){
        qDebug() << "ScatterMatrix sizes not equal for legendreOrder="<<getLegendreOrder()<<" temperature="<<getTemp();
        return false;
    }
    for( int i = 0; i < sinkGroups.size(); i++){
        bool equal = *sinkGroups[i] == *a.sinkGroups[i];
        if( !equal ){
            qDebug() << "ScatterMatrix sinkgroup not equal for legendreOrder="<<getLegendreOrder()<<" temperature="<<getTemp()<<" at index "<<i;
            return false;
        }
    }
    if( legendreOrder != a.legendreOrder ){
        qDebug() << "ScatterMatrix legendreOrder not equal for legendreOrder="<<getLegendreOrder()<<" temperature="<<getTemp();
        return false;
    }
    if( temp != a.temp ){
        qDebug() << "ScatterMatrix temperature not equal for legendreOrder="<<getLegendreOrder()<<" temperature="<<getTemp()<<" vs a.l="<<a.getLegendreOrder()<<" a.t="<<a.getTemp();
        return false;
    }
    return true;
}
void ScatterMatrix::initialize()
{
    legendreOrder = 0;
    temp = 0;
}
ScatterMatrix::~ScatterMatrix()
{
    for( int i = 0; i < sinkGroups.size();i++){
        SinkGroup * sinkGroup = sinkGroups.value(i);
        delete sinkGroup;
    }
    sinkGroups.clear();
}

ScatterMatrix * ScatterMatrix::getCopy() const
{
    ScatterMatrix * copy = new ScatterMatrix(*this);
    return copy;
}

bool ScatterMatrix::getFirstLastGroup(int & first, int & last)
{
    if( sinkGroups.size() == 0 ) return false;
    first = std::numeric_limits<int>::max();
    last = std::numeric_limits<int>::min();
    for( int i = 0; i < sinkGroups.size(); i++ ){
        SinkGroup * sink = sinkGroups.at(i);
        if( sink->getGrp() < first ) first = sink->getGrp();
        if( sink->getGrp() > last ) last = sink->getGrp();
    }
    return true;
}
float * ScatterMatrix::getCollapsedMatrix(int & length)
{
    float * data = NULL;
    length = 0; /// make sure initialize to zero
    QList<float> floats;
    /// deal with special case where no sinkgroups returns
    /// array of length 2, where [0]=mgw=-1
    if( sinkGroups.size() == 0 ){
        length = 2;
        floats.append(-1.0);
        floats.append(0.0);
        ///cerr<<"***Warning: ScatterMatrix has no SinkGroups!"<<endl
    }
    /// traverse sinkgroups aggregating data
    for( int i = 0 ; i < sinkGroups.size(); i++)
    {
        SinkGroup * sink = sinkGroups.value(i);
        // make sure to reverse data back from logical start to end to Ampx end to start
        sink->reverseData();
        /// construct magic word for sink group
        bool error = false;
        FloatIntUnion mw;
        mw.magic_word = getMagicWord(sink->getGrp(), sink->getStart(), sink->getEnd(), error);
        if( error ){
            cerr<<"***Error: MagicWord generation failed!"
                <<endl
                <<"          grp="<<sink->getGrp()<<", start="<<sink->getStart()<<", end="<<sink->getEnd()
                <<endl;
            return NULL;
        }
        // Reinterpret integer magic word as a floating point number.
        floats.append(mw.magic_float);
        /// add start to end data
        for( int j = sink->getStart(); j < sink->getEnd(); j++)
            floats.append(sink->get(j));
        sink->reverseData();

    }
    length = floats.size();
    data = new float[length];
    for( int i = 0; i < length; i++)
        data[i] = floats.value(i);
    return data;
}

// Serialization interfaces

const long ScatterMatrix::uid = 0xabcdef0000000000;



void quickSort(QList<ScatterMatrix*> &list)
{
    if( list.size() <= 1 ) return;

    int pivotIndex = list.size()/2;
    if( pivotIndex == 1 ){
        if( !(*list[0] < *list[1]) ) list.swap(0,1);
        return;
    }
    QList<ScatterMatrix*> left, right;
    ScatterMatrix * pivot = list[pivotIndex];
    for( int i = 0; i < list.size() ; i ++){
        if( i == pivotIndex ) continue;
        if( *list[i] < *pivot ){
            left.append(list[i]);
        }else{
            right.append(list[i]);
        }
    }

    //Ensure(left.size() + right.size() + 1== list.size());
    quickSort(left);
    quickSort(right);

    //Ensure(left.size() + right.size() + 1== list.size());
    for( int i = 0; i < left.size(); i ++){
        list[i]= left[i];
    }
    list[left.size()]=pivot;
    for( int i = 0, l=left.size()+1; i < right.size(); i ++, l++){
        //Ensure(l < list.size());
        //Ensure(i < right.size());
        list[l]= right[i];
    }
}
