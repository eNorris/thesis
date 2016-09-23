//#include "Standard/Interface/AbstractStream.h"
//#include "Resource/AmpxLib/ampxlib_config.h"
#include "LibraryNuclide.h"
#include "BondarenkoGlobal.h"
#include "NuclideResonance.h"
#include "AmpxLibrary.h"
#include <string>
#include <QDebug>
#include "AmpxDataHelper.h"

LibraryNuclide::LibraryNuclide() {
    initialize();
}

LibraryNuclide::LibraryNuclide( const LibraryNuclide & orig, NuclideFilter * f){
    d = orig.d;
    d->ref.ref();
    bondarenkoGlobalData = NULL;
    subGroups = NULL;
    if( orig.subGroups  != NULL) subGroups = orig.subGroups->getCopy();


    setResonance(NULL);
    setTotal2dXS(NULL);
    NuclideFilter filter(f!=NULL ? *f : NuclideFilter(NuclideFilter::Keep)); // default global policy is Keep
    AmpxLibrary * library = filter.getAmpxLib();
    qDebug() << qPrintable("Nuclide ")<<getId()<<" filtered copy with library at address ";
    if (orig.totalXS != NULL) {
        //qDebug()<<"Copying total xs...";
        //if( filter.acceptsTotal() )
        this->setTotal2dXS(orig.totalXS->getCopy());
    }
    if( orig.resonance != NULL){
        this->resonance = new NuclideResonance(*orig.resonance);
    }
    this->setNumBondarenkoSig0Data(0);
    this->setNumBondarenkoTempData(0);
    this->setMaxNumBondGrp(0);
    this->setNumBondarenkoData(0);
    if (orig.bondarenkoGlobalData != NULL)
        for (int i = 0; i < orig.bondarenkoData.size(); i++){
            if( filter.acceptsBondarenko(orig.bondarenkoData.at(i)->getMt() ) ) {
                this->bondarenkoGlobalData = orig.bondarenkoGlobalData->getCopy();
                this->setNumBondarenkoSig0Data(const_cast<LibraryNuclide*>(&orig)->getNumBondarenkoSig0Data());
                this->setNumBondarenkoTempData(const_cast<LibraryNuclide*>(&orig)->getNumBondarenkoTempData());
                this->setMaxNumBondGrp(const_cast<LibraryNuclide*>(&orig)->getMaxNumBondGrp());
                break;
            }
        }
        //if( filter.acceptsBondarenkoGlobal() )
            //this->bondarenkoGlobalData = orig.bondarenkoGlobalData->getCopy();
    //qDebug()<<"Done copying global...";

    if( library != NULL )
    {
        char * description = AmpxDataHelper::getDescription(const_cast<LibraryNuclide*>(&orig), library);
        this->setDescription(description);
    }
    else
    {
        for( int i = 0; i < AMPX_NUCLIDE_DESCR_LENGTH+1; i++)
            this->d->description[i] = orig.d->description[i];
        if( this->getDescription() == NULL ) {
            //for( int i = 0;i < AMPX_NUCLIDE_DESCR_LENGTH; i++)
        //char achar=' ';
        for( int i = 0; i < AMPX_NUCLIDE_DESCR_LENGTH; i++)
            this->d->description[i] = ' ';
        }
        this->d->description[AMPX_NUCLIDE_DESCR_LENGTH] = '\0';
    }
    // The following copies are based on the contents of the appropriate lists,
    // not the number counts stored in the orig information directory
    if( library != NULL )
    {
        QList<int> * mts = AmpxDataHelper::getBondarenkoDataMts(const_cast<LibraryNuclide*>(&orig), library);
        for( int i = 0; i < mts->size(); i++ )
        {
            if( filter.acceptsBondarenko(mts->at(i) ) ){
                this->bondarenkoData.append(AmpxDataHelper::getBondarenkoDataByMt(const_cast<LibraryNuclide*>(&orig),library,mts->at(i),true)->getCopy());
                this->setNumBondarenkoData(this->bondarenkoData.size());
            }
        }
        delete mts;
    }
    else
    {
        for (int i = 0; i < orig.bondarenkoData.size(); i++){
            if( filter.acceptsBondarenko(orig.bondarenkoData.at(i)->getMt() ) ){
                qDebug() << qPrintable("Accepting bondarenko ")<<orig.bondarenkoData.at(i)->getMt();
                this->bondarenkoData.append(orig.bondarenkoData.at(i)->getCopy());
                this->setNumBondarenkoData(this->bondarenkoData.size());
            }else
            {
                qDebug() << qPrintable("NOT Accepting bondarenko ")<<orig.bondarenkoData.at(i)->getMt();
            }
        }
    }
    if( library != NULL )
    {
        QList<int> * mts = AmpxDataHelper::getNeutron1dXSMts(const_cast<LibraryNuclide*>(&orig), library);
        for( int i = 0; i < mts->size(); i++ )
        {
            if( filter.acceptsNeutron1d(mts->at(i) ) ){
                this->neutron1dData.append(AmpxDataHelper::getNeutron1dXSByMt(const_cast<LibraryNuclide*>(&orig),library,mts->at(i),true)->getCopy());
                this->setNumNeutron1dData(this->neutron1dData.size());
            }
        }
        delete mts;
    }
    else
    {
        for (int i = 0; i < orig.neutron1dData.size(); i++){
            if( filter.acceptsNeutron1d(orig.neutron1dData.at(i)->getMt() ) ){
                this->neutron1dData.append(orig.neutron1dData.at(i)->getCopy());
                this->setNumNeutron1dData(this->neutron1dData.size());
            }
        }
    }
    if( library != NULL )
    {
        QList<int> * mts = AmpxDataHelper::getGamma1dXSMts(const_cast<LibraryNuclide*>(&orig), library);
        for( int i = 0; i < mts->size(); i++ )
        {
            if( filter.acceptsGamma1d(mts->at(i) ) ){
                this->gamma1dData.append(AmpxDataHelper::getGamma1dXSByMt(const_cast<LibraryNuclide*>(&orig),library,mts->at(i),true)->getCopy());
                this->setNumGamma1dData(this->gamma1dData.size());
            }
        }
        delete mts;
    }
    else
    {
        for (int i = 0; i < orig.gamma1dData.size(); i++){
            if( filter.acceptsGamma1d(orig.gamma1dData.at(i)->getMt() ) ){
                this->gamma1dData.append(orig.gamma1dData.at(i)->getCopy());
                this->setNumGamma1dData(this->gamma1dData.size());
            }
        }
    }
    if( library != NULL )
    {
        QList<int> * mts = AmpxDataHelper::getNeutron2dXSMts(const_cast<LibraryNuclide*>(&orig), library);
        for( int i = 0; i < mts->size(); i++ )
        {
            if( filter.acceptsNeutron2d(mts->at(i) ) )
            {
                this->neutron2dData.append(AmpxDataHelper::getNeutron2dXSByMt(const_cast<LibraryNuclide*>(&orig),library,mts->at(i),true)->getCopy());
                this->setNumNeutron2dData(this->neutron2dData.size());
            }
        }
        delete mts;
    }
    else
    {
        for (int i = 0; i < orig.neutron2dData.size(); i++){
            if( filter.acceptsNeutron2d(orig.neutron2dData.at(i)->getMt() ) ){
                this->neutron2dData.append(orig.neutron2dData.at(i)->getCopy());
                this->setNumNeutron2dData(this->neutron2dData.size());
            }
        }
    }
    if( library != NULL )
    {
        QList<int> * mts = AmpxDataHelper::getGamma2dXSMts(const_cast<LibraryNuclide*>(&orig), library);
        for( int i = 0; i < mts->size(); i++ )
        {
            if( filter.acceptsGamma2d(mts->at(i) ) ){
                this->gamma2dData.append(AmpxDataHelper::getGamma2dXSByMt(const_cast<LibraryNuclide*>(&orig),library,mts->at(i),true)->getCopy());
                this->setNumGamma2dData(this->gamma2dData.size());
            }
        }
        delete mts;
    }
    else
    {
        for (int i = 0; i < orig.gamma2dData.size(); i++){
            if( filter.acceptsGamma2d(orig.gamma2dData.at(i)->getMt() ) ){
                this->gamma2dData.append(orig.gamma2dData.at(i)->getCopy());
                this->setNumGamma2dData(this->gamma2dData.size());
            }
        }
    }
    if( library != NULL )
    {
        QList<int> * mts = AmpxDataHelper::getGammaProdXSMts(const_cast<LibraryNuclide*>(&orig), library);
        for( int i = 0; i < mts->size(); i++ )
        {
            if( filter.acceptsGammaProd(mts->at(i) ) ){
                this->gammaProdData.append(AmpxDataHelper::getGammaProdXSByMt(const_cast<LibraryNuclide*>(&orig),library,mts->at(i),true)->getCopy());
                this->setNumGammaProdData(this->gammaProdData.size());
            }
        }
        delete mts;
    }
    else
    {
        for (int i = 0; i < orig.gammaProdData.size(); i++){
            if( filter.acceptsGammaProd(orig.gammaProdData.at(i)->getMt() ) ){
                this->gammaProdData.append(orig.gammaProdData.at(i)->getCopy());
                this->setNumGammaProdData(this->gammaProdData.size());
            }
        }
    }

}

void LibraryNuclide::initialize() {
    d = new LibraryNuclideData();
    subGroups = NULL;
    setBondarenkoGlobal(NULL);
    setResonance(NULL);
    setTotal2dXS(NULL);
}// end of initialize

LibraryNuclide::~LibraryNuclide() {
    if( !d->ref.deref()){
        delete d;
    }
    if( this->bondarenkoGlobalData != NULL )
    {
        delete this->bondarenkoGlobalData;
    }
    if( this->totalXS != NULL ) delete this->totalXS;

    for (int i = 0; i < bondarenkoData.size(); i++)
        delete bondarenkoData.value(i);
    bondarenkoData.clear();
    for (int i = 0; i < neutron1dData.size(); i++)
        delete neutron1dData.value(i);
    neutron1dData.clear();
    for (int i = 0; i < neutron2dData.size(); i++)
        delete neutron2dData.value(i);
    neutron2dData.clear();
    for (int i = 0; i < gamma1dData.size(); i++)
        delete gamma1dData.value(i);
    gamma1dData.clear();
    for (int i = 0; i < gamma2dData.size(); i++)
        delete gamma2dData.value(i);
    gamma2dData.clear();
    for (int i = 0; i < gammaProdData.size(); i++)
        delete gammaProdData.value(i);
    gammaProdData.clear();
    if (getResonance() != NULL)
        delete getResonance();

    if( subGroups != NULL) delete subGroups;
}

bool LibraryNuclide::operator==(LibraryNuclide & a){
    if( bondarenkoGlobalData != NULL && a.bondarenkoGlobalData != NULL ){
        bool equal = *bondarenkoGlobalData == *a.bondarenkoGlobalData;
        if( !equal )
        {
            qDebug() << qPrintable("BondarenkoGlobal not equal for nuclide id=")<<getId();
            return false;
        }
    }else if(bondarenkoGlobalData == NULL && a.bondarenkoGlobalData != NULL ){
        qDebug() << qPrintable("BondarenkoGlobal not equal for nuclide id=")<<getId();
        return false;
    }else if(bondarenkoGlobalData != NULL && a.bondarenkoGlobalData == NULL ){
        qDebug() << qPrintable("BondarenkoGlobal not equal for nuclide id=")<<getId();
        return false;
    }
    if( totalXS != NULL && a.totalXS != NULL ){
        bool equal = *totalXS == *a.totalXS;
        qDebug() << qPrintable("totalXS2d equal ")<<equal;
        if( !equal ){
            qDebug() << qPrintable("TotalXS not equal for nuclide id=")<<getId();
            return false;
        }
    }else if(totalXS == NULL && a.totalXS != NULL ){
        qDebug() << qPrintable("TotalXS not equal for nuclide id=")<<getId();
        return false;
    }else if(totalXS != NULL && a.totalXS == NULL ){
        qDebug() << qPrintable("TotalXS not equal for nuclide id=")<<getId();
        return false;
    }
    if( bondarenkoData.size() != a.bondarenkoData.size() ){
        qDebug() << qPrintable("BondarenkoData of different sizes ")<<bondarenkoData.size()<<" "<<a.bondarenkoData.size();
        return false;
    }

    for( int i = 0; i < bondarenkoData.size(); i++){
        bool equal = *bondarenkoData[i] == *a.bondarenkoData[i];
         if( !equal ) qDebug() << qPrintable("BondarenkoData at index ")<<i<<" equal "<<equal;
        if( !equal ) return false;
    }


    if( subGroups != 0 && a.subGroups != 0){
        bool equal = *subGroups == *a.subGroups;
        if( !equal ) qDebug() << qPrintable("Subgroup data do not agree for nuclide id=")<<getId();
        if( !equal) return false;
    }
    else{
        if( subGroups != a.subGroups ){
            return false;
        }
    }

    if( neutron1dData.size() != a.neutron1dData.size() ){
        qDebug() << qPrintable("Neutron1d of different sizes ")<<neutron1dData.size()<<" "<<a.neutron1dData.size();
        return false;
    }

    for( int i = 0; i < neutron1dData.size(); i++){
        bool equal = *neutron1dData[i] == *a.neutron1dData[i];
        if( !equal )
        {
            qDebug() << qPrintable("Neutron1d not equal for nuclide id=")<<getId()<<" at index "<<i;
            return false;
        }
    }
    if( gamma1dData.size() != a.gamma1dData.size() ){
        qDebug() << qPrintable("Gamma1d of different sizes ")<<gamma1dData.size()<<" "<<a.gamma1dData.size();
        return false;
    }

    for( int i = 0; i < gamma1dData.size(); i++){
        bool equal = *gamma1dData[i] == *a.gamma1dData[i];
        if( !equal ){
            qDebug() << qPrintable("Gamma1d not equal for nuclide id=")<<getId()<<" at index "<<i;
            return false;
        }
    }
    if( neutron2dData.size() != a.neutron2dData.size() ){
        qDebug() << qPrintable("Neutron2d of different sizes ")<<bondarenkoData.size()<<" "<<a.bondarenkoData.size();
        return false;
    }

    for( int i = 0; i < neutron2dData.size(); i++){
        bool equal = *neutron2dData[i] == *a.neutron2dData[i];
        if( !equal )
        {
            qDebug() << qPrintable("Neutron2d not equal for nuclide id=")<<getId()<<" at index "<<i;
            return false;
        }
    }
    if( gamma2dData.size() != a.gamma2dData.size() ){
        qDebug() << qPrintable("Gamma2d of different sizes ")<<gamma2dData.size()<<" "<<a.gamma2dData.size();
        return false;
    }

    for( int i = 0; i < gamma2dData.size(); i++){
        bool equal = *gamma2dData[i] == *a.gamma2dData[i];
        if( !equal ){
            qDebug() << qPrintable("Gamma2d not equal for nuclide id=")<<getId()<<" at index "<<i;
            return false;
        }
    }
    if( gammaProdData.size() != a.gammaProdData.size() ){
        qDebug() << qPrintable("GammaProd of different sizes ")<<gammaProdData.size()<<" "<<a.gammaProdData.size();
        return false;
    }

    for( int i = 0; i < gammaProdData.size(); i++){
        bool equal = *gammaProdData[i] == *a.gammaProdData[i];
        if( !equal ){
            qDebug() << qPrintable("GammaProd not equal for nuclide id=")<<getId()<<" at index "<<i;
            return false;
        }
    }
    if( d == a.d ) return true;
    if( strcmp(d->description, a.d->description) != 0 ){
        qDebug() << qPrintable("Nuclide description different '")<<d->description<<"' vs '"<<a.d->description<<"'.";
        return false;
    }
    for( int i = 0; i < AMPX_NUMBER_NUCLIDE_INTEGER_OPTIONS; i++){
        if( d->words[i] != a.d->words[i] ){
            qDebug() << qPrintable("Nuclide word ")<<i<<" is different "<<d->words[i]<<" "<<a.d->words[i];
            return false;
        }
    }



    return true;
}
bool LibraryNuclide::containsBondarenkoDataByMt(int mt) {
    for( int i = 0; i < bondarenkoData.size(); i ++){
      if( bondarenkoData[i]->getMt() == mt ) return true;
    }
    return false;
}

bool LibraryNuclide::containsNeutron1dDataByMt(int mt) {
    for( int i = 0; i < neutron1dData.size(); i ++){
      if( neutron1dData[i]->getMt() == mt ) return true;
    }
    return false;
}

bool LibraryNuclide::containsNeutron2dDataByMt(int mt) {
    for( int i = 0; i < neutron2dData.size(); i ++){
      if( neutron2dData[i]->getMt() == mt ) return true;
    }
    return false;
}

bool LibraryNuclide::containsGamma1dDataByMt(int mt) {
    for( int i = 0; i < gamma1dData.size(); i ++){
      if( gamma1dData[i]->getMt() == mt ) return true;
    }
    return false;
}

bool LibraryNuclide::containsGamma2dDataByMt(int mt) {
    for( int i = 0; i < gamma2dData.size(); i ++){
      if( gamma2dData[i]->getMt() == mt ) return true;
    }
    return false;
}

bool LibraryNuclide::containsGammaProdDataByMt(int mt) {
    for( int i = 0; i < gammaProdData.size(); i ++){
      if( gammaProdData[i]->getMt() == mt ) return true;
    }
    return false;
}

LibraryNuclide * LibraryNuclide::getCopy() const {
    LibraryNuclide * copy = new LibraryNuclide(*this);
    return copy;
}
LibraryNuclide * LibraryNuclide::getCopy(NuclideFilter& filter) {
    LibraryNuclide * copy = new LibraryNuclide(*this,&filter);
    return copy;
}

void LibraryNuclide::copyFrom(LibraryNuclide & nuclide, bool fixAsMaster) {
    //qDebug()<<"Copying nuclide...";
    /// copy nuclides data words
    qAtomicAssign(d, nuclide.d);
    // fix as master is used when you are copying a working nuclide
    // to a library nuclide and you want to fix the nuclide directory
    if( fixAsMaster ){
        for( int i = 2; i <= 9; i++)
            setData(i,0);
        for( int i = 17; i <=22; i++)
            setData(i,0);

        /// deep copy
        if (nuclide.getTotal2dXS() != NULL) {
            //qDebug()<<"Copying total xs...";
            this->setTotal2dXS(nuclide.getTotal2dXS()->getCopy());
        }
        if (nuclide.getBondarenkoGlobal() != NULL)
            this->setBondarenkoGlobal(nuclide.getBondarenkoGlobal()->getCopy());
        //qDebug()<<"Done copying global...";

        // The following copies are based on the contents of the appropriate lists,
        // not the number counts stored in the nuclide information directory
        for (int i = 0; i < nuclide.bondarenkoData.size(); i++)
            this->addBondarenkoData(nuclide.getBondarenkoDataAt(i)->getCopy());
        //qDebug()<<"Done copying bondata...";
        for (int i = 0; i < nuclide.neutron1dData.size(); i++)
            this->addNeutron1dData(nuclide.getNeutron1dDataAt(i)->getCopy());
        //qDebug()<<"Done copying neutron1d...";
        for (int i = 0; i < nuclide.gamma1dData.size(); i++)
            this->addGamma1dData(nuclide.getGamma1dDataAt(i)->getCopy());
        //qDebug()<<"Done copying gamma1d...";
        for (int i = 0; i < nuclide.neutron2dData.size(); i++)
            this->addNeutron2dData(nuclide.getNeutron2dDataAt(i)->getCopy());
        //qDebug()<<"Done copying neutron2d...";
        for (int i = 0; i < nuclide.gamma2dData.size(); i++)
            this->addGamma2dData(nuclide.getGamma2dDataAt(i)->getCopy());
        //qDebug()<<"Done copying gamma2d...";
        for (int i = 0; i < nuclide.gammaProdData.size(); i++)
            this->addGammaProdData(nuclide.getGammaProdDataAt(i)->getCopy());
        //qDebug()<<"Done copying gammaProd...";
    }else{
        if (nuclide.getTotal2dXS() != NULL) {
            //qDebug()<<"Copying total xs...";
            this->setTotal2dXS(nuclide.getTotal2dXS()->getCopy());
        }
        if (nuclide.getBondarenkoGlobal() != NULL)
            this->bondarenkoGlobalData = nuclide.getBondarenkoGlobal()->getCopy();
        //qDebug()<<"Done copying global...";

        // The following copies are based on the contents of the appropriate lists,
        // not the number counts stored in the nuclide information directory
        for (int i = 0; i < nuclide.bondarenkoData.size(); i++)
            this->bondarenkoData.append(nuclide.getBondarenkoDataAt(i)->getCopy());
        //qDebug()<<"Done copying bondata...";
        for (int i = 0; i < nuclide.neutron1dData.size(); i++)
            this->neutron1dData.append(nuclide.getNeutron1dDataAt(i)->getCopy());
        //qDebug()<<"Done copying neutron1d...";
        for (int i = 0; i < nuclide.gamma1dData.size(); i++)
            this->gamma1dData.append(nuclide.getGamma1dDataAt(i)->getCopy());
        //qDebug()<<"Done copying gamma1d...";
        for (int i = 0; i < nuclide.neutron2dData.size(); i++)
            this->neutron2dData.append(nuclide.getNeutron2dDataAt(i)->getCopy());
        //qDebug()<<"Done copying neutron2d...";
        for (int i = 0; i < nuclide.gamma2dData.size(); i++)
            this->gamma2dData.append(nuclide.getGamma2dDataAt(i)->getCopy());
        //qDebug()<<"Done copying gamma2d...";
        for (int i = 0; i < nuclide.gammaProdData.size(); i++)
            this->gammaProdData.append(nuclide.getGammaProdDataAt(i)->getCopy());
    }
    if( nuclide.resonance != NULL){
        this->resonance = new NuclideResonance(*nuclide.resonance);
    }
}

void LibraryNuclide::setId(int id) {
    int oldid = getId();
    if (id == oldid) return;
    setData(AMPX_MAST_NUCLIDE_ID, id);
}

void LibraryNuclide::setMixture(int mix) {
    int oldMixId = getMixture();
    if( mix == oldMixId ) return;
    setData(AMPX_NUCLIDE_MIXTURE, mix);
}

// Serialization interfaces

const long LibraryNuclide::uid = 0xfbacf55248146501;

/**
 * @brief Serialize the object into a contiguous block of data
 * @param AbstractStream * stream - the stream into which the contiguous data will be stored
 * @return int - 0 upon success, error otherwise
 */
/*
int LibraryNuclide::serialize(Standard::AbstractStream * stream) const{
    if( stream == NULL ) return -1;
    long classUID = this->getUID();
    stream->write((char*)&classUID, sizeof(classUID));

    // write the size of this object
    unsigned long serializedSize = getSerializedSize();
    stream->write((char*)&serializedSize, sizeof(serializedSize));

    // DBC=4 - capture the oldWriteHead for checksumming later
    Remember(long oldWriteHead = stream->getWriteHead());

// directory information (description and 50 word record)
    stream->write((char*)d->description, AMPX_NUCLIDE_DESCR_LENGTH);
    stream->write((char*)d->words, sizeof(int)*AMPX_NUMBER_NUCLIDE_INTEGER_OPTIONS);

    bool globalPresent = bondarenkoGlobalData != NULL;
    stream->write((char*)&globalPresent, sizeof(globalPresent));
    if( globalPresent ){
        int rtncode = bondarenkoGlobalData->serialize(stream);
        if( rtncode != 0) return rtncode;
    }

    bool totalXS2dPresent = totalXS != NULL;
    stream->write((char*)&totalXS2dPresent, sizeof(totalXS2dPresent));
    if( totalXS2dPresent ){
        int rtncode = totalXS->serialize(stream);
        if( rtncode != 0) return rtncode;
    }
    int bondarenkoCount = bondarenkoData.size();
    stream->write((char*)&bondarenkoCount, sizeof(bondarenkoCount));
    for( int i = 0; i < bondarenkoCount; i++){
        int rtncode = bondarenkoData[i]->serialize(stream);
        if( rtncode != 0 ) return rtncode;
    }
    int neutron1dCount = neutron1dData.size();
    stream->write((char*)&neutron1dCount, sizeof(neutron1dCount));
    for( int i = 0; i < neutron1dCount; i++){
        int rtncode = neutron1dData[i]->serialize(stream);
        if( rtncode != 0 ) return rtncode;
    }
    int gamma1dCount = gamma1dData.size();
    stream->write((char*)&gamma1dCount, sizeof(gamma1dCount));
    for( int i = 0; i < gamma1dCount; i++){
        int rtncode = gamma1dData[i]->serialize(stream);
        if( rtncode != 0 ) return rtncode;
    }
    int neutron2dCount = neutron2dData.size();
    stream->write((char*)&neutron2dCount, sizeof(neutron2dCount));
    for( int i = 0; i < neutron2dCount; i++){
        int rtncode = neutron2dData[i]->serialize(stream);
        if( rtncode != 0 ) return rtncode;
    }
    int gamma2dCount = gamma2dData.size();
    stream->write((char*)&gamma2dCount, sizeof(gamma2dCount));
    for( int i = 0; i < gamma2dCount; i++){
        int rtncode = gamma2dData[i]->serialize(stream);
        if( rtncode != 0 ) return rtncode;
    }
    int gammaProdCount = gammaProdData.size();
    stream->write((char*)&gammaProdCount, sizeof(gammaProdCount));
    for( int i = 0; i < gammaProdCount; i++){
        int rtncode = gammaProdData[i]->serialize(stream);
        if( rtncode != 0 ) return rtncode;
    }

     // subgroup data
    bool subPresent = subGroups != NULL;
    stream->write((char*)&subPresent, sizeof(subPresent));
    if( subPresent ){
        int rtncode = subGroups->serialize(stream);
        if( rtncode != 0) return rtncode;
    }

    // DBC=4 - checksum the expected serialized size and the actual serialized size
    Ensure( static_cast<unsigned long>(stream->getWriteHead() - oldWriteHead) == LibraryNuclide::getSerializedSize() );
    return 0;
}
*/

/**
 * @brief deserialize the object from the given AbstractStream
 * @param AbstractStream * stream - the stream from which the object will be inflated
 * @return int - 0 upon success, error otherwise
 */
/*
int LibraryNuclide::deserialize(Standard::AbstractStream * stream){
    long read_uid = stream->getNextUID();

    // make sure we know how to parse the object in the stream
    if( read_uid != this->getUID() ) return Serializable::TypeMismatch;

    // skip over uid
    stream->ignore(sizeof(read_uid));

    // read objects serialized size
    unsigned long serializedSize;
    stream->read((char*)&serializedSize, sizeof(serializedSize));

    // need to make sure data is only owned by this object
    qAtomicDetach(d);

    // directory information (description and 50 word record)
    stream->read((char*)d->description, AMPX_NUCLIDE_DESCR_LENGTH);
    d->description[AMPX_NUCLIDE_DESCR_LENGTH] = '\0'; // null terminate the string
    stream->read((char*)d->words, sizeof(int)*AMPX_NUMBER_NUCLIDE_INTEGER_OPTIONS);

    bool globalPresent;
    stream->read((char*)&globalPresent, sizeof(globalPresent));
    if( globalPresent ){
        if( !stream->isNext(BondarenkoGlobal::uid) ){
            qDebug() << qPrintable("Failed to find BondarenkoGlobal data as the next object... ChildMistmatch error!");
            return Serializable::ChildTypeMismatch;
        }
        Serializable * obj = stream->deserializeNext();
        if( obj == NULL ){
            qDebug() << qPrintable("Failed to deserialize the BondarenkoGlobal object!");
            return Serializable::FailedObjectDeserialization;
        }
        bondarenkoGlobalData = (BondarenkoGlobal * ) obj;
    }

    bool totalXS2dPresent;
    stream->read((char*)&totalXS2dPresent, sizeof(totalXS2dPresent));
    if( totalXS2dPresent ){
        if( !stream->isNext(CrossSection2d::uid) ){
            qDebug() << qPrintable("Failed to find total XS2d data as the next object... ChildMistmatch error!");
            return Serializable::ChildTypeMismatch;
        }
        Serializable * obj = stream->deserializeNext();
        if( obj == NULL ){
            qDebug() << qPrintable("Failed to deserialize the BondarenkoGlobal object!");
            return Serializable::FailedObjectDeserialization;
        }
        totalXS = (CrossSection2d * ) obj;
    }
    int bondarenkoDataCount;
    stream->read((char*)&bondarenkoDataCount, sizeof(bondarenkoDataCount));
    for( int i = 0; i < bondarenkoDataCount; i++){
        if( !stream->isNext(BondarenkoData::uid) ){
            qDebug() << qPrintable("Failed to find  bondarenkoData data as the next object... ChildMistmatch error!");
            return Serializable::ChildTypeMismatch;
        }
        Serializable * obj = stream->deserializeNext();
        if( obj == NULL ){
            qDebug() << qPrintable("Failed to deserialize the next bondarenkoData object!");
            return Serializable::FailedObjectDeserialization;
        }
        bondarenkoData.append((BondarenkoData * ) obj);
    }
    int neutron1dCount;
    stream->read((char*)&neutron1dCount, sizeof(neutron1dCount));
    for( int i = 0; i < neutron1dCount; i++){
        if( !stream->isNext(CrossSection1d::uid) ){
            qDebug() << qPrintable("Failed to find  neutron1d data as the next object... ChildMistmatch error!");
            return Serializable::ChildTypeMismatch;
        }
        Serializable * obj = stream->deserializeNext();
        if( obj == NULL ){
            qDebug() << qPrintable("Failed to deserialize the next neutron1d object!");
            return Serializable::FailedObjectDeserialization;
        }
        neutron1dData.append((CrossSection1d * ) obj);
    }
    int gamma1dCount;
    stream->read((char*)&gamma1dCount, sizeof(gamma1dCount));
    for( int i = 0; i < gamma1dCount; i++){
        if( !stream->isNext(CrossSection1d::uid) ){
            qDebug() << qPrintable("Failed to find  gamma1d data as the next object... ChildMistmatch error!");
            return Serializable::ChildTypeMismatch;
        }
        Serializable * obj = stream->deserializeNext();
        if( obj == NULL ){
            qDebug() << qPrintable("Failed to deserialize the next gamma1d object!");
            return Serializable::FailedObjectDeserialization;
        }
        gamma1dData.append((CrossSection1d * ) obj);
    }
    int neutron2dCount;
    stream->read((char*)&neutron2dCount, sizeof(neutron2dCount));
    for( int i = 0; i < neutron2dCount; i++){
        if( !stream->isNext(CrossSection2d::uid) ){
            qDebug() << qPrintable("Failed to find  neutron2d data as the next object... ChildMistmatch error!");
            return Serializable::ChildTypeMismatch;
        }
        Serializable * obj = stream->deserializeNext();
        if( obj == NULL ){
            qDebug() << qPrintable("Failed to deserialize the next neutron2d object!");
            return Serializable::FailedObjectDeserialization;
        }
        neutron2dData.append((CrossSection2d * ) obj);
    }
    int gamma2dCount;
    stream->read((char*)&gamma2dCount, sizeof(gamma2dCount));
    for( int i = 0; i < gamma2dCount; i++){
        if( !stream->isNext(CrossSection2d::uid) ){
            qDebug() << qPrintable("Failed to find  gamma2d data as the next object... ChildMistmatch error!");
            return Serializable::ChildTypeMismatch;
        }
        Serializable * obj = stream->deserializeNext();
        if( obj == NULL ){
            qDebug() << qPrintable("Failed to deserialize the next gamma2d object!");
            return Serializable::FailedObjectDeserialization;
        }
        gamma2dData.append((CrossSection2d * ) obj);
    }
    int gammaProdCount;
    stream->read((char*)&gammaProdCount, sizeof(gammaProdCount));    for( int i = 0; i < gammaProdCount; i++){
        if( !stream->isNext(CrossSection2d::uid) ){
            qDebug() << qPrintable("Failed to find  gammaProd data as the next object... ChildMistmatch error!");
            return Serializable::ChildTypeMismatch;
        }
        Serializable * obj = stream->deserializeNext();
        if( obj == NULL ){
            qDebug() << qPrintable("Failed to deserialize the next gammaProd object!");
            return Serializable::FailedObjectDeserialization;
        }
        gammaProdData.append((CrossSection2d * ) obj);
    }


    // subgroup data
    bool subPresent;
    stream->read((char*)&subPresent, sizeof(subPresent));
    this->setSubGroupData(0);
    if( subPresent)  {
        if( !stream->isNext(SubGroupData::uid) ){
            qDebug() << qPrintable("Failed to find  SubGroupData in the next object... ChildMistmatch error!");
            return Serializable::ChildTypeMismatch;
        }
        Serializable * obj = stream->deserializeNext();
        if( obj == NULL ){
            qDebug() << qPrintable("Failed to deserialize the next SubGroupData object!");
            return Serializable::FailedObjectDeserialization;
        }
        this->setSubGroupData( (SubGroupData*) obj);
    }

    return 0;
}
*/
/**
 * @brief Obtain the size in bytes of this object when serialized
 * @return unsigned long - the size in bytes of the object when serialized
 */
/*
unsigned long LibraryNuclide::getSerializedSize() const{
    unsigned long size = 0;

    // directory information (description and 50 word record)
    size = AMPX_NUCLIDE_DESCR_LENGTH + sizeof(int)*AMPX_NUMBER_NUCLIDE_INTEGER_OPTIONS;

    // bondarenko Global data
    bool globalPresent = bondarenkoGlobalData != NULL;
    size += sizeof(globalPresent);
    if( globalPresent ){
        unsigned long serialized = bondarenkoGlobalData->getSerializedSize();
        size += serialized;
        size += sizeof(bondarenkoGlobalData->getUID());
        size += sizeof(serialized);
    }
    // totalXS2d data
    bool totalXS2dPresent = totalXS != NULL;
    size += sizeof(totalXS2dPresent);
    if( totalXS2dPresent ){
        unsigned long serialized = totalXS->getSerializedSize();
        size += serialized;
        size += sizeof(totalXS->getUID());
        size += sizeof(serialized);
    }
    // bondarenko data
    size += sizeof(bondarenkoData.size());
    for( int i = 0; i < bondarenkoData.size(); i++){
        size += sizeof(bondarenkoData[i]->getUID());
        unsigned long serialized = bondarenkoData[i]->getSerializedSize();
        size += serialized;
        size += sizeof(serialized);
    }

    // neutron1d data
    size += sizeof(neutron1dData.size());
    for( int i = 0; i < neutron1dData.size(); i++){
        size += sizeof(neutron1dData[i]->getUID());
        unsigned long serialized = neutron1dData[i]->getSerializedSize();
        size += serialized;
        size += sizeof(serialized);
    }
    // gamma1d data
    size += sizeof(gamma1dData.size());
    for( int i = 0; i < gamma1dData.size(); i++){
        size += sizeof(gamma1dData[i]->getUID());
        unsigned long serialized = gamma1dData[i]->getSerializedSize();
        size += serialized;
        size += sizeof(serialized);
    }
    // neutron2d data
    size += sizeof(neutron2dData.size());
    for( int i = 0; i < neutron2dData.size(); i++){
        size += sizeof(neutron2dData[i]->getUID());
        unsigned long serialized = neutron2dData[i]->getSerializedSize();
        size += serialized;
        size += sizeof(serialized);
    }
    // gamma2d data
    size += sizeof(gamma2dData.size());
    for( int i = 0; i < gamma2dData.size(); i++){
        size += sizeof(gamma2dData[i]->getUID());
        unsigned long serialized = gamma2dData[i]->getSerializedSize();
        size += serialized;
        size += sizeof(serialized);
    }
    // gammaProd data
    size += sizeof(gammaProdData.size());
    for( int i = 0; i < gammaProdData.size(); i++){
        size += sizeof(gammaProdData[i]->getUID());
        unsigned long serialized = gammaProdData[i]->getSerializedSize();
        size += serialized;
        size += sizeof(serialized);
    }

    // subgroup data
    bool subPresent = subGroups != NULL;
    size += sizeof(subPresent);
    if( subPresent ){
        unsigned long serialized = subGroups->getSerializedSize();
        size += serialized;
        size += sizeof(subGroups->getUID());
        size += sizeof(serialized);
    }

    return size;
}
*/
/**
 * @brief Obtain the universal version identifier
 * the uid should be unique for all Serializable objects such that
 * an object factory can retrieve prototypes of the desired Serializable
 * object and inflate the serial object into a manageable object
 * @return long - the uid of the object
 */
long LibraryNuclide::getUID() const{
    return LibraryNuclide::uid;
}
