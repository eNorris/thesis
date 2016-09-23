#include "SubGroupData.h"
//#include "Nemesis/harness/DBC.hh"
//#include "Standard/Interface/jdebug.h"

bool SubGroupWeight::operator==(SubGroupWeight & a){
    if( temp != a.temp) return false;
    CrossSection1d *c1,*c2;
    c1 = (CrossSection1d *)this;
    c2 = (CrossSection1d *)&a;
    return *c1 == *c2;
}



const long SubGroupWeight::uid = 0xbaeef95102041612;

SubGroupSetGrp::SubGroupSetGrp(const SubGroupSetGrp& orig){
    ig = orig.ig;

    for( int i = 0; i < orig.cross.size(); i++)
        this->cross.append(orig.cross[i]->getCopy());

     for( int i = 0; i < orig.weight.size(); i++)
        this->weight.append(orig.weight[i]->getCopy());
}


SubGroupSetGrp::~SubGroupSetGrp(){
    for( int i = 0; i < cross.size(); i++){
        delete cross.value(i);
    }
    cross.clear();
    for( int i = 0; i < weight.size(); i++){
        delete weight.value(i);
    }
    weight.clear();
}

void  SubGroupSetGrp::addTemperatures(QList<float> * temps) const{
    if( temps == 0) return;
    for( int i = 0; i < weight.size(); i++ ) {
        SubGroupWeight * w = weight.at(i);
        if( !temps->contains(w->getTemp()) ) temps->append(w->getTemp());
    }
    qSort(*temps);
}

void SubGroupSetGrp::addMts(QList<int> * mts) const{
    if( mts == 0) return;
    for( int i = 0; i < weight.size(); i++ ) {
        SubGroupWeight * w = weight.at(i);
        if( !mts->contains(w->getMt()) ) mts->append(w->getMt());
    }

    for( int i = 0; i < cross.size(); i++ ) {
        CrossSection1d * w = cross.at(i);
        if( !mts->contains(w->getMt()) ) mts->append(w->getMt());
    }

    qSort(*mts);
}



CrossSection1d * SubGroupSetGrp::getCrossByMt(int mt){
    for( int i = 0; i < cross.size(); i++ ) {
        if( cross[i]->getMt() == mt) return cross[i];
    }
    return 0;
}

SubGroupWeight *  SubGroupSetGrp::getWeightByMt(int mt, float temp){
    for( int i = 0; i < weight.size(); i++ ) {
        if( weight[i]->getMt() == mt && weight[i]->getTemp() == temp) return weight[i];
    }
    return 0;
}


 bool SubGroupSetGrp::containsCrossByMt(int mt){
    for( int i = 0; i < cross.size(); i++ ) {
        if( cross[i]->getMt() == mt) return true;
    }
    return false;
 }

 bool SubGroupSetGrp::operator==(SubGroupSetGrp & a){
     if( this->ig != a.ig) return false;
     for( int i = 0; i < cross.size(); i++){
        CrossSection1d  * c = a.getCrossByMt(cross[i]->getMt());
        if( c == 0) return false;
        bool tmp = *c == *cross[i];
        if( !tmp) return false;
     }
     for( int i = 0; i < a.cross.size(); i++){
        CrossSection1d  * c = this->getCrossByMt(a.cross[i]->getMt());
        if( c == 0) return false;
        bool tmp = *c == *cross[i];
        if( !tmp) return false;
     }

     for( int i = 0; i < weight.size(); i++){
        SubGroupWeight  * c = a.getWeightByMt(weight[i]->getMt(),weight[i]->getTemp());
        if( c == 0) return false;
        bool tmp = *c == *weight[i];
        if( !tmp) return false;
     }
     for( int i = 0; i < a.weight.size(); i++){
        SubGroupWeight  * c = this->getWeightByMt(a.weight[i]->getMt(),a.weight[i]->getTemp());
        if( c == 0) return false;
        bool tmp = *c == *weight[i];
        if( !tmp) return false;
     }

     return true;
 }


const long SubGroupSet::uid = 0xbafcf96113441812;

SubGroupSet::SubGroupSet(const SubGroupSet& orig){
    groups = 0;
    this->ngrp = orig.ngrp;
    if (orig.groups != 0) {
        groups = new SubGroupSetGrp*[ngrp];
        for (int i = 0; i < ngrp; i++) {
            groups[i] = 0;
            if (orig.groups[i] != 0) groups[i] = new SubGroupSetGrp(*orig.groups[i]);
        }
    }
}

SubGroupSet::~SubGroupSet() {
    if (groups != 0) {
        for (int i = 0; i < ngrp; i++) {
            if (groups[i] != 0) delete groups[i];
        }
        delete [] groups;
    }
    groups = 0;
}

void SubGroupSet::setSize(int size){
    if( size < 0) return;
    if( ngrp == size) return;
    if( size == 0){
        for (int i = 0; i < ngrp; i++) {
            if (groups[i] != 0) delete groups[i];
        }
        delete [] groups;
        ngrp = 0;
        return;
    }

    SubGroupSetGrp **tmp = 0;
    tmp = new SubGroupSetGrp*[size];
    int i,mm;
    mm = ngrp;
    if( mm > size) mm = size;
    for (i = 0; i < mm; i++) tmp[i] = groups[i];
    for( i = mm; i < ngrp; i++) {
        if( groups[i] != 0) delete groups[i];
    }
    for( i = mm; i < size; i++) tmp[i] = 0;
    if( groups != 0) delete [] groups;
    groups = tmp;

    ngrp = size;
}

void SubGroupSet::addMts(QList<int> * mts) const{
    if( mts == 0) return;
    for( int i = 0; i < ngrp; i++ ) {
        if( groups[i] == 0) continue;
        groups[i]->addMts(mts);
    }
    qSort(*mts);
}


void  SubGroupSet::addTemperatures(QList<float> * temps) const{
    if( temps == 0) return;
    for( int i = 0; i < ngrp; i++ ) {
        if( groups[i] == 0) continue;
        groups[i]->addTemperatures(temps);
    }
    qSort(*temps);
}

CrossSection1d * SubGroupSet::getCrossByMt(int mt, int ig){
    for( int i = 0; i < ngrp; i++){
        if( groups[i] != 0 && groups[i]->getGrp() == ig){
            return groups[i]->getCrossByMt(mt);
        }
    }
    return 0;
}


SubGroupWeight * SubGroupSet::getWeightByMt(int mt, float temp, int ig){
     for( int i = 0; i < ngrp; i++){
        if( groups[i] != 0 && groups[i]->getGrp() == ig){
            return groups[i]->getWeightByMt(mt,temp);
        }
    }
    return 0;
}

bool SubGroupSet::operator==(SubGroupSet & a){
    if( this->ngrp != a.ngrp) return false;
    for( int i = 0; i < ngrp; i++){
        if( groups[i] == 0){
            if( a.groups[i] != 0) return false;
        }
        else{
            if( a.groups[i] == 0) return false;
             bool tmp = *groups[i] == *a.groups[i];
             if( !tmp) return false;
        }
    }
    return true;
 }





SubGroupData::SubGroupData(const SubGroupData& orig) {
    neutStart = orig.neutStart;
    neutEnd = orig.neutEnd;
    for(int i = 0; i < orig.subGrps.size();i++){
        if( orig.subGrps[i] != 0) subGrps.append(new SubGroupSet(*orig.subGrps[i]));
    }
}

SubGroupData::~SubGroupData() {
    for(int i = 0; i < subGrps.size();i++){
        if( subGrps[i] != 0) delete subGrps[i];
    }
}

const long SubGroupData::uid = 0xbafaf96743441913;


bool SubGroupData::operator==(SubGroupData & a){
    if( this->neutStart != a.neutStart)  return false;
    if( this->neutEnd != a.neutEnd) return false;



    for( int i = 0; i < subGrps.size(); i++){
        bool found = false;
        for( int j = 0; j < a.subGrps.size(); j++){
            if( subGrps[i]->getSize() == a.subGrps[j]->getSize()){
                found = true;
                bool tmp = *subGrps[i] == *a.subGrps[j];
                if( !tmp) return false;
            }
        }
        if( !found) return false;
    }

     for( int i = 0; i < a.subGrps.size(); i++){
        bool found = false;
        for( int j = 0; j < subGrps.size(); j++){
            if( a.subGrps[i]->getSize() == subGrps[j]->getSize()){
                found = true;
                bool tmp = *a.subGrps[i] == *subGrps[j];
                if( !tmp) return false;
            }
        }
        if( !found) return false;
    }
    return true;
}



