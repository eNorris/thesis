//#include "Nemesis/harness/DBC.hh"
#include "SinkGroup.h"
#include "resources.h"
#include <iostream>
#include <QDebug>
//#include "Standard/Interface/AbstractStream.h"
//#include "Resource/AmpxLib/ampxlib_config.h"
using namespace std;

SinkGroup::SinkGroup()
{
    initialize();
}
SinkGroup::SinkGroup(const SinkGroup & orig){
    start = orig.start;
    end = orig.end;
    grp = orig.grp;
    data = orig.data;
}
const SinkGroup& SinkGroup::operator=(const SinkGroup & orig){

    data = orig.data;
    start = orig.start;
    end = orig.end;
    grp = orig.grp;
    return *this;
}
bool SinkGroup::operator==(SinkGroup & a){
    if( data.size() != a.data.size())
    {
        qDebug() << "SinkGroup sizes not equal for grp="<<getGrp()<<" s="<<getStart()<<" e="<<getEnd()<<" vs a.grp="<<a.getGrp()<<" a.s="<<a.getStart()<<" a.e="<<a.getEnd();
        return false;
    }
    for( int i = 0; i < data.size(); i++){
        if( data[i] != a.data[i] )
        {
            qDebug() << "SinkGroup data not equal for grp"<<getGrp()<<" s="<<getStart()<<" e="<<getEnd()<<" at index "<<i<<" "<<data[i]<<" vs "<<a.data[i];
            return false;
        }
    }
    //if( this->d == d) return true;
    if( start != a.start )
    {
        qDebug() << "SinkGroup start not equal for grp"<<getGrp()<<" s="<<getStart()<<" e="<<getEnd();
        return false;
    }
    if( end != a.end ){
        qDebug() << "SinkGroup end not equal for grp"<<getGrp()<<" s="<<getStart()<<" e="<<getEnd();
        return false;
    }
    if( grp != a.grp )
    {
        qDebug() << "SinkGroup grp not equal for grp"<<getGrp()<<" s="<<getStart()<<" e="<<getEnd();
        return false;
    }
    return true;
}
void SinkGroup::initialize()
{
    start = 0;
    end = 0;
    grp = 0;
}
SinkGroup::~SinkGroup()
{
}

void SinkGroup::reverseData()
{
    for( int i =0, j = data.size()-1; i<j; i++, j--){
        float temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
}
SinkGroup * SinkGroup::getCopy() const
{
    SinkGroup * copy = new SinkGroup(*this);    
    return copy;
}

QList<SinkGroup*> SinkGroup::expandMagicWordArray(float * data, int length, bool verbose)
{
    QList<SinkGroup*> set;
    if( data == NULL ) return set;
    if( length < 2 ) return set;
    int grp=-1, start=0, end=0;
    int index = 0;
    do{
        int sinkLength = 0;
        /// we expect the first 4 byte word to be a integer magic word
        int mgcWord = *reinterpret_cast<int*>(&data[index]);
        if( !explodeMagicWord(mgcWord, grp, start, end, sinkLength) ){
            break;
        }
        index++;
        if(verbose) cout<<"Magic word, grp="<<grp<<", start="<<start<<", end="<<end<<", length="<<sinkLength<<endl;
        if( sinkLength <= 0 ) break;
        SinkGroup * sinkGroup = new SinkGroup();
        sinkGroup->data.reserve(sinkLength);
        sinkGroup->setGrp(grp);
        sinkGroup->setStart(start);
        sinkGroup->setEnd(end);
        for( int i = 0; i < sinkLength; i++, index++ ){
            sinkGroup->set(start+i,data[index]);
            if(verbose) cout<<"sink["<<grp<<"]["<<start+i<<"]="<<sinkGroup->get(start+i)<<endl;
        }
        sinkGroup->reverseData(); // make sure to reversedata from values end to start, to more logical start to end
        set.append(sinkGroup);
    }while( grp!= -1 && index < length );
    return set;
} // end of expandMagicWordArray

// serializable interface
const long SinkGroup::uid = 0x09000000f0b00777;

