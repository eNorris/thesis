/*
 * File:   AmpxDataHelper.cpp
 * Author: raq
 *
 * Created on April 2, 2012, 1:36 PM
 */

#include "AmpxDataHelper.h"
#include "CrossSection2d.h"
#include "CrossSection1d.h"
#include "AmpxLibrary.h"
#include "LibraryNuclide.h"
#include <QDebug>
//#include "Resource/AmpxLib/ampxlib_config.h"
AmpxDataHelper::AmpxDataHelper() {
}

AmpxDataHelper::AmpxDataHelper(const AmpxDataHelper& orig) {
}

AmpxDataHelper::~AmpxDataHelper() {
}
/**
    * @brief Obtain the nuclide description
    * @param nuclide - the nuclide for which to obtain the NumRecords
    * @param library - the library that backs up the nuclide
    * @return char * - the character array containing the description
    * Note: This checks the nuclide to determine if the nuclide has the description
    * if the nuclide's description is NULL, it attempts to retrieve the value from the nuclide
    * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably NULL) from the
    * given nuclide will be returned.
    */
char * AmpxDataHelper::getDescription(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);
    char * description = nuclide->getDescription();
    if( description !=NULL ){
        return description;
    }
    if( library == NULL ) return description;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return description;
    return libraryNuclide->getDescription();
}
/**
    * @brief Obtain the NumRecords
    * @param nuclide - the nuclide for which to obtain the NumRecords
    * @param library - the library that backs up the nuclide
    * @return int - the NumRecords
    * Note: This check the nuclide to determine if the nuclide has the NumRecords
    * if the nuclide's NumRecords is zero, it attempts to retrieve the value from the nuclide
    * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
    * given nuclide will be returned.
    */
int AmpxDataHelper::getNumRecords(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);
    int numRecords = nuclide->getNumRecords();
    if( numRecords != 0 ){
        return numRecords;
    }
    if( library == NULL ) return numRecords;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return numRecords;
    nuclide->setNumRecords(libraryNuclide->getNumRecords());
    return nuclide->getNumRecords();
}
/**
    * @brief Obtain the GammaProdData
    * @param nuclide - the nuclide for which to obtain the GammaProdData
    * @param library - the library that backs up the nuclide
    * @return int - the GammaProdData
    * Note: This check the nuclide to determine if the nuclide has the GammaProdData
    * if the nuclide's GammaProdData is zero, it attempts to retrieve the value from the nuclide
    * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
    * given nuclide will be returned.
    */
int AmpxDataHelper::getGammaProdData(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);
    int gammaProdData = nuclide->getGammaProdData();
    if( gammaProdData != 0 ){
        return gammaProdData;
    }
    if( library == NULL ) return gammaProdData;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return gammaProdData;
    nuclide->setGammaProdData(libraryNuclide->getGammaProdData());
    return nuclide->getGammaProdData();
}
/**
    * @brief Obtain the GammaData
    * @param nuclide - the nuclide for which to obtain the GammaData
    * @param library - the library that backs up the nuclide
    * @return int - the GammaData
    * Note: This check the nuclide to determine if the nuclide has the GammaData
    * if the nuclide's GammaData is zero, it attempts to retrieve the value from the nuclide
    * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
    * given nuclide will be returned.
    */
int AmpxDataHelper::getGammaData(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);
    int gammaData = nuclide->getGammaData();
    if( gammaData != 0 ){
        return gammaData;
    }
    if( library == NULL ) return gammaData;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return gammaData;
    nuclide->setGammaData(libraryNuclide->getGammaData());
    return nuclide->getGammaData();
}
/**
    * @brief Obtain the ThermalNeutronData
    * @param nuclide - the nuclide for which to obtain the ThermalNeutronData
    * @param library - the library that backs up the nuclide
    * @return int - the ThermalNeutronData
    * Note: This check the nuclide to determine if the nuclide has the ThermalNeutronData
    * if the nuclide's ThermalNeutronData is zero, it attempts to retrieve the value from the nuclide
    * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
    * given nuclide will be returned.
    */
int AmpxDataHelper::getThermalNeutronData(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);
    int thermalNeutronData = nuclide->getThermalNeutronData();
    if( thermalNeutronData != 0 ){
        return thermalNeutronData;
    }
    if( library == NULL ) return thermalNeutronData;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return thermalNeutronData;
    nuclide->setThermalNeutronData(libraryNuclide->getThermalNeutronData());
    return nuclide->getThermalNeutronData();
}
/**
    * @brief Obtain the FastNeutronData
    * @param nuclide - the nuclide for which to obtain the FastNeutronData
    * @param library - the library that backs up the nuclide
    * @return int - the FastNeutronData
    * Note: This check the nuclide to determine if the nuclide has the FastNeutronData
    * if the nuclide's FastNeutronData is zero, it attempts to retrieve the value from the nuclide
    * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
    * given nuclide will be returned.
    */
int AmpxDataHelper::getFastNeutronData(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);
    int fastNeutronData = nuclide->getFastNeutronData();
    if( fastNeutronData != 0 ){
        return fastNeutronData;
    }
    if( library == NULL ) return fastNeutronData;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return fastNeutronData;
    nuclide->setFastNeutronData(libraryNuclide->getFastNeutronData());
    return nuclide->getFastNeutronData();
}
/**
    * @brief Obtain the PotentialScatter
    * @param nuclide - the nuclide for which to obtain the PotentialScatter
    * @param library - the library that backs up the nuclide
    * @return float - the PotentialScatter
    * Note: This check the nuclide to determine if the nuclide has the PotentialScatter
    * if the nuclide's PotentialScatter is zero, it attempts to retrieve the value from the nuclide
    * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
    * given nuclide will be returned.
    */
float AmpxDataHelper::getPotentialScatter(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);
    float potentialScatter = nuclide->getPotentialScatter();
    if( potentialScatter != 0.0 ){
        return potentialScatter;
    }
    if( library == NULL ) return potentialScatter;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return potentialScatter;
    nuclide->setPotentialScatter(libraryNuclide->getPotentialScatter());
    return nuclide->getPotentialScatter();
}
/**
    * @brief Obtain the MaxLengthScatterArray
    * @param nuclide - the nuclide for which to obtain the MaxLengthScatterArray
    * @param library - the library that backs up the nuclide
    * @return int - the MaxLengthScatterArray
    * Note: This check the nuclide to determine if the nuclide has the MaxLengthScatterArray
    * if the nuclide's MaxLengthScatterArray is zero, it attempts to retrieve the value from the nuclide
    * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
    * given nuclide will be returned.
    */
int AmpxDataHelper::getMaxLengthScatterArray(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);
    int maxLengthScatterArray = nuclide->getMaxLengthScatterArray();
    if( maxLengthScatterArray != 0 ){
        return maxLengthScatterArray;
    }
    if( library == NULL ) return maxLengthScatterArray;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return maxLengthScatterArray;
    nuclide->setMaxLengthScatterArray(libraryNuclide->getMaxLengthScatterArray());
    return nuclide->getMaxLengthScatterArray();
}
/**
    * @brief Obtain the EnergyReleasePerCapture
    * @param nuclide - the nuclide for which to obtain the EnergyReleasePerCapture
    * @param library - the library that backs up the nuclide
    * @return float - the EnergyReleasePerCapture
    * Note: This check the nuclide to determine if the nuclide has the EnergyReleasePerCapture
    * if the nuclide's EnergyReleasePerCapture is zero, it attempts to retrieve the value from the nuclide
    * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
    * given nuclide will be returned.
    */
float AmpxDataHelper::getEnergyReleasePerCapture(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);
    float energyReleasePerCapture = nuclide->getEnergyReleasePerCapture();
    if( energyReleasePerCapture != 0 ){
        return energyReleasePerCapture;
    }
    if( library == NULL ) return energyReleasePerCapture;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return energyReleasePerCapture;
    nuclide->setEnergyReleasePerCapture(libraryNuclide->getEnergyReleasePerCapture());
    return nuclide->getEnergyReleasePerCapture();
}
/**
    * @brief Obtain the PowerPerFission
    * @param nuclide - the nuclide for which to obtain the PowerPerFission
    * @param library - the library that backs up the nuclide
    * @return float - the PowerPerFission
    * Note: This check the nuclide to determine if the nuclide has the PowerPerFission
    * if the nuclide's PowerPerFission is zero, it attempts to retrieve the value from the nuclide
    * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
    * given nuclide will be returned.
    */
float AmpxDataHelper::getPowerPerFission(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);
    float powerPerFission = nuclide->getPowerPerFission();
    if( powerPerFission != 0.0 ){
        return powerPerFission;
    }
    if( library == NULL ) return powerPerFission;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return powerPerFission;
    nuclide->setPowerPerFission(libraryNuclide->getPowerPerFission());
    return nuclide->getPowerPerFission();
}
/**
    * @brief Obtain the ZA
    * @param nuclide - the nuclide for which to obtain the ZA
    * @param library - the library that backs up the nuclide
    * @return int - the ZA
    * Note: This check the nuclide to determine if the nuclide has the ZA
    * if the nuclide's ZA is zero, it attempts to retrieve the value from the nuclide
    * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
    * given nuclide will be returned.
    */
int AmpxDataHelper::getZA(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);
    int za = nuclide->getZA();
    if( za != 0 ){
        return za;
    }
    if( library == NULL ) return za;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return za;
    nuclide->setZA(libraryNuclide->getZA());
    return nuclide->getZA();
}
/**
    * @brief Obtain the AtomicMass
    * @param nuclide - the nuclide for which to obtain the AtomicMass
    * @param library - the library that backs up the nuclide
    * @return float - the AtomicMass
    * Note: This check the nuclide to determine if the nuclide has the AtomicMass
    * if the nuclide's AtomicMass is zero, it attempts to retrieve the value from the nuclide
    * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
    * given nuclide will be returned.
    */
float AmpxDataHelper::getAtomicMass(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);
    float atomicMass = nuclide->getAtomicMass();
    if( atomicMass != 0.0 ){
        return atomicMass;
    }
    if( library == NULL ) return atomicMass;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return atomicMass;
    nuclide->setAtomicMass(libraryNuclide->getAtomicMass());
    return nuclide->getAtomicMass();
}

/**
    * @brief Obtain the NumEnergiesUnresolvedResonance
    * @param nuclide - the nuclide for which to obtain the NumEnergiesUnresolvedResonance
    * @param library - the library that backs up the nuclide
    * @return int - the NumEnergiesUnresolvedResonance
    * Note: This check the nuclide to determine if the nuclide has the NumEnergiesUnresolvedResonance
    * if the nuclide's NumEnergiesUnresolvedResonance is zero, it attempts to retrieve the value from the nuclide
    * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
    * given nuclide will be returned.
    */
int AmpxDataHelper::getNumEnergiesUnresolvedResonance(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);
    int numEnergiesUnresolvedResonance = nuclide->getNumEnergiesUnresolvedResonance();
    if( numEnergiesUnresolvedResonance != 0 ){
        return numEnergiesUnresolvedResonance;
    }
    if( library == NULL ) return numEnergiesUnresolvedResonance;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return numEnergiesUnresolvedResonance;
    nuclide->setNumEnergiesUnresolvedResonance(libraryNuclide->getNumEnergiesUnresolvedResonance());
    return nuclide->getNumEnergiesUnresolvedResonance();
}
/**
    * @brief Obtain the NumResolvedResonance
    * @param nuclide - the nuclide for which to obtain the NumResolvedResonance
    * @param library - the library that backs up the nuclide
    * @return int - the NumResolvedResonance
    * Note: This check the nuclide to determine if the nuclide has the NumResolvedResonance
    * if the nuclide's NumResolvedResonance is zero, it attempts to retrieve the value from the nuclide
    * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
    * given nuclide will be returned.
    */
int AmpxDataHelper::getNumResolvedResonance(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);
    int numResolvedResonance = nuclide->getNumResolvedResonance();
    if( numResolvedResonance != 0 ){
        return numResolvedResonance;
    }
    if( library == NULL ) return numResolvedResonance;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return numResolvedResonance;
    nuclide->setNumResolvedResonance(libraryNuclide->getNumResolvedResonance());
    return nuclide->getNumResolvedResonance();
}
/**
    * @brief Obtain the MaxNumBondGrp
    * @param nuclide - the nuclide for which to obtain the MaxNumBondGrp
    * @param library - the library that backs up the nuclide
    * @return int - the MaxNumBondGrp
    * Note: This check the nuclide to determine if the nuclide has the MaxNumBondGrp
    * if the nuclide's MaxNumBondGrp is zero, it attempts to retrieve the value from the nuclide
    * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
    * given nuclide will be returned.
    */
int AmpxDataHelper::getMaxNumBondGrp(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);
    int maxNumBondGrp = nuclide->getMaxNumBondGrp();
    if( maxNumBondGrp != 0 ){
        return maxNumBondGrp;
    }
    if( library == NULL ) return maxNumBondGrp;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return maxNumBondGrp;
    nuclide->setMaxNumBondGrp(libraryNuclide->getMaxNumBondGrp());
    return nuclide->getMaxNumBondGrp();
}
/**
    * @brief Obtain the global bondarenko data for the given nuclide
    * @param nuclide - the nuclide from which to obtain the global bondarenko data
    * @param library - the library that backs up the nuclide
    * @param readOnly - indicator of whether or not the caller intends to write/ modify the data
    * @return BondarenkoGlobal * - BondarenkoGlobal data, or null if none was found on either the nuclide or the library
    * Note: If the data is not found on the nuclide, but is found on the library and the readOnly=false, then the global data
    * will be copied onto the nuclide for modification and future retrieval.
    */
BondarenkoGlobal* AmpxDataHelper::getBondarenkoGlobalData(LibraryNuclide* nuclide, AmpxLibrary * library, bool readOnly){
    //Require(nuclide != NULL );

    BondarenkoGlobal* bg = NULL; // ensure that the bg is NULL

    // determine if the nuclide already has it
    BondarenkoGlobal* nuclideBg = nuclide->getBondarenkoGlobal();
    if( nuclideBg != NULL ){
        qDebug() << "Found global bondarenko data on nuclide "<<nuclide->getId();
        bg = nuclideBg;
        return bg;
    }

    if( library == NULL ) return bg; // return if no library was specified

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return NULL;

    BondarenkoGlobal * global = libraryNuclide->getBondarenkoGlobal();

    if( global == NULL ) return NULL;

    if( readOnly ){
        qDebug() << "Found global bondarenko data on library";
        return global;
    }

    BondarenkoGlobal * globalCopy = global->getCopy();
    //Check( globalCopy != NULL );
    nuclide->setBondarenkoGlobal(globalCopy);
    qDebug() << "Found global bondarenko data on library and placed copy on nuclide "<<nuclide->getId();
    return globalCopy;
}
/**
    * @brief Obtain the list of 1d mts that are available for the given nuclide
    * @param nuclide - the nuclide for which to obtain the list of mts
    * @param library - the ampx library that backs up the nuclide
    * @return QList<int> * - a newly allocated list of nuclide ids (THIS WILL NEED TO BE DELETED BY YOU)
    */
QList<int> * AmpxDataHelper::getBondarenkoDataMts(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);

    QList<int> * mts = new QList<int>();

    //Check( mts != NULL );

    // obtain the mts from the nuclide
    for( int i = 0; i < nuclide->getNumBondarenkoData(); i++){
        BondarenkoData * bd = nuclide->getBondarenkoDataAt(i);
        //Check( bd != NULL );
        mts->append(bd->getMt());
    }

    // obtain the mts from the library's nuclide
    if( library == NULL ) return mts; // return if no library was specified
    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return mts; // return if no nuclide is present on the library

    // obtain the mts from the nuclide
    for( int i = 0; i < libraryNuclide->getNumBondarenkoData(); i++){
        BondarenkoData * bd = libraryNuclide->getBondarenkoDataAt(i);
        //Check( bd != NULL );
        mts->append(bd->getMt());
    }

    // sort in preparation for duplicate mt removal
    qSort(*mts);
    // remove duplicates
    for( int i = mts->size() -1; i >= 1; i-- ){
        if( mts->at(i) == mts->at(i-1) ) mts->removeAt(i);
    }
    return mts;
}
/**
    * @brief Obtain the bondarenko data 1d with the given mt from either the nuclide or the ampx library
    * This method obtains the bondarenko data 1d with the given mt from the nuclide or the library. The method
    * check the nuclide first, and if the nuclide does not contain the given mt, it attempts to retrieve the bondarenko data
    * from the library.
    *
    * If the bondarenko data is retrieved from the library and readOnly==false, the bondarenko data is copied onto
    * the nuclide, such that any writes will not effect the original library bondarenko data.
    * @param nuclide - the nuclide from which the bondarenko data will come from or be placed on if not present
    * @param library - the library from which the bondarenko data will be obtained if not on the nuclide
    * @param mt - the type of bondarenko data to be obtained
    * @param readOnly - if the bondarenko data is obtained from the library, this indicates if the library needs to be copied onto the nuclide
    * @return BondarenkoData* - the desired bondarenko datas obtained from the nuclide or the library, NULL if none were available
    */
BondarenkoData* AmpxDataHelper::getBondarenkoDataByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt, bool readOnly ){
    //Require( nuclide != NULL );

    BondarenkoData * bd = NULL; // ensure that the bd is NULL

    // determine if the nuclide already has the desired cross section
    BondarenkoData * nuclideBd = nuclide->getBondarenkoDataByMt(mt);
    if( nuclideBd != NULL ){
        bd = nuclideBd; // set the desired cross section
        return bd;
    }

    if( library == NULL ) return bd; // return if no library was specified

    // determine if the library contains the nuclide and the desired cross section
    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());

    if( libraryNuclide == NULL ) return NULL;
    // determine if the library nuclide has the desired cross section
    BondarenkoData * libraryNuclideBd = libraryNuclide->getBondarenkoDataByMt(mt);
    if( libraryNuclideBd == NULL )  return NULL;

    // if we are read only
    // we have found fulfilled the caller's request
    if( readOnly ){
        bd = libraryNuclideBd;
        return bd;
    }

    // if we are here, than we will need to copy the cross sections
    // because the caller has indicated they will be writing the data
    // and we do not want the original library cross sections to be changed
    BondarenkoData * writableBd = libraryNuclideBd->getCopy();
    //Check( writableBd != NULL );

    // add the library cross section to the nuclide so
    // writes will be captured in the future
    bool added = nuclide->addBondarenkoData(writableBd);
    //Insist( added, "Was not added." );
    bd = writableBd;
    return bd;
}

/**
    * @brief Obtain the list of mts that are available for the given nuclide
    * @param nuclide - the nuclide for which to obtain the list of mts
    * @param library - the ampx library that backs up the nuclide
    * @return QList<int> * - a newly allocated list of nuclide ids (THIS WILL NEED TO BE DELETED BY YOU)
    */
QList<int> * AmpxDataHelper::getNeutron1dXSMts(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);

    QList<int> * mts = new QList<int>();

    //Check( mts != NULL );

    // obtain the mts from the nuclide
    for( int i = 0; i < nuclide->getNeutron1dList().size(); i++){
        CrossSection1d * xs1d = nuclide->getNeutron1dDataAt(i);
        //Check( xs1d != NULL );
        mts->append(xs1d->getMt());
    }
    int mtCount = mts->size();
    // obtain the mts from the library's nuclide
    if( library == NULL ) return mts; // return if no library was specified
    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return mts; // return if no nuclide is present on the library

        // obtain the mts from the nuclide
    for( int i = 0; i < libraryNuclide->getNeutron1dList().size(); i++){
        CrossSection1d * xs1d = libraryNuclide->getNeutron1dDataAt(i);
        //Check( xs1d != NULL );
        mts->append(xs1d->getMt());
    }

    // sort in preparation for duplicate mt removal
    qSort(*mts);
    // remove duplicates
    for( int i = mts->size() -1; i >= 1; i-- ){
        if( mts->at(i) == mts->at(i-1) ) mts->removeAt(i);
    }
    qDebug() << "MT total count "<<mts->size()<<" nuclide count "<<mtCount;
    return mts;
}
/**
    * @brief Obtain the neutron cross section 1d with the given mt from either the nuclide or the ampx library
    * This method obtains the cross section 1d with the given mt from the nuclide or the library. The method
    * check the nuclide first, and if the nuclide does not contain the given mt, it attempts to retrieve the cross section
    * from the library.
    *
    * If the cross section is retrieved from the library and readOnly==false, the cross section is copied onto
    * the nuclide, such that any writes will not effect the original library cross section.
    * @param nuclide - the nuclide from which the cross section will come from or be placed on if not present
    * @param library - the library from which the cross section will be obtained if not on the nuclide
    * @param mt - the type of cross section to be obtained
    * @param readOnly - if the cross section is obtained from the library, this indicates if the library needs to be copied onto the nuclide
    * @param CrossSection2d - the desired cross sections obtained from the nuclide or the library, NULL if none were available
    */
CrossSection1d * AmpxDataHelper::getNeutron1dXSByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt, bool readOnly ){
    //Require( nuclide != NULL );
    CrossSection1d * xs = NULL;
    // determine if the nuclide already has the desired cross section
    CrossSection1d * nuclideXs = nuclide->getNeutron1dDataByMt(mt);
    if( nuclideXs != NULL ){
        qDebug() << "Found neutron1d xs "<<mt<<" on given nuclide";
        xs = nuclideXs; // set the desired cross section
        return xs;
    }

    if( library == NULL ) return xs; // return if no library was specified

    // determine if the library contains the nuclide and the desired cross section
    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());

    if( libraryNuclide == NULL ){
        qDebug() << "Failed to find nuclide on library with id "<<nuclide->getId();
        return NULL;
    }
    // determine if the library nuclide has the desired cross section
    CrossSection1d * libraryNuclideXs = libraryNuclide->getNeutron1dDataByMt(mt);
    if( libraryNuclideXs == NULL ){
        //qDebug() << ("Failed to find neutron1d xs by mt "<<mt<<" on either nuclide or library");
        return NULL;
    }

    // if we are read only
    // we have found fulfilled the caller's request
    if( readOnly ){
        //qDebug() << ("Found neutron1d xs "<<mt<<" on given ampxlibrary");
        xs = libraryNuclideXs;
        return xs;
    }

    // if we are here, than we will need to copy the cross sections
    // because the caller has indicated they will be writing the data
    // and we do not want the original library cross sections to be changed
    CrossSection1d * writableXs = libraryNuclideXs->getCopy();
    //Check( writableXs != NULL );

    // add the library cross section to the nuclide so
    // writes will be captured in the future
    bool added = nuclide->addNeutron1dData(writableXs);
    //Insist( added, "Was not added." );
    xs = writableXs;
    qDebug() << "Found neutron1d xs "<<mt<<" on given ampxlibrary and made copy for writing purposes";
    return xs;
}
/**
    * @brief Obtain the list of mts that are available for the given nuclide
    * @param nuclide - the nuclide for which to obtain the list of mts
    * @param library - the ampx library that backs up the nuclide
    * @return QList<int> * - a newly allocated list of nuclide ids (THIS WILL NEED TO BE DELETED BY YOU)
    */
QList<int> * AmpxDataHelper::getNeutron2dXSMts(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);

    QList<int> * mts = new QList<int>();

    //Check( mts != NULL );

    // obtain the mts from the nuclide
    for( int i = 0; i < nuclide->getNeutron2dList().size(); i++){
        CrossSection2d * xs2d = nuclide->getNeutron2dDataAt(i);
        //Check( xs2d != NULL );
        mts->append(xs2d->getMt());
    }

    // obtian the mts from the library's nuclide
    if( library == NULL ) return mts; // return if no library was specified
    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return mts; // return if no nuclide is present on the library

        // obtain the mts from the nuclide
    for( int i = 0; i < libraryNuclide->getNeutron2dList().size(); i++){
        CrossSection2d * xs2d = libraryNuclide->getNeutron2dDataAt(i);
        //Check( xs2d != NULL );
        mts->append(xs2d->getMt());
    }

    // sort in preparation for duplicate mt removal
    qSort(*mts);
    // remove duplicates
    for( int i = mts->size() -1; i >= 1; i-- ){
        if( mts->at(i) == mts->at(i-1) ) mts->removeAt(i);
    }
    return mts;
}
/**
    * @brief Obtain the neutron cross section 2d with the given mt from either the nuclide or the ampx library
    * This method obtains the cross section 2d with the given mt from the nuclide or the library. The method
    * check the nuclide first, and if the nuclide does not contain the given mt, it attempts to retrieve the cross section
    * from the library.
    *
    * If the cross section is retrieved from the library and readOnly==false, the cross section is copied onto
    * the nuclide, such that any writes will not effect the original library cross section.
    * @param nuclide - the nuclide from which the cross section will come from or be placed on if not present
    * @param library - the library from which the cross section will be obtained if not on the nuclide
    * @param mt - the type of cross section to be obtained
    * @param readOnly - if the cross section is obtained from the library, this indicates if the library needs to be copied onto the nuclide
    * @param CrossSection2d - the desired cross sections obtained from the nuclide or the library, NULL if none were available
    */
CrossSection2d * AmpxDataHelper::getNeutron2dXSByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt, bool readOnly){
    //Require( nuclide != NULL );

    CrossSection2d * xs = NULL; // ensure that the xs is NULL

    // determine if the nuclide already has the desired cross section
    CrossSection2d * nuclideXs = nuclide->getNeutron2dDataByMt(mt);
    if( nuclideXs != NULL ){
        xs = nuclideXs; // set the desired cross section
        return xs;
    }

    if( library == NULL ) return xs; // return if no library was specified

    // determine if the library contains the nuclide and the desired cross section
    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());

    if( libraryNuclide == NULL ) return NULL;
    // determine if the library nuclide has the desired cross section
    CrossSection2d * libraryNuclideXs = libraryNuclide->getNeutron2dDataByMt(mt);
    if( libraryNuclideXs == NULL )  return NULL;

    // if we are read only
    // we have found fulfilled the caller's request
    if( readOnly ){
        xs = libraryNuclideXs;
        return xs;
    }

    // if we are here, than we will need to copy the cross sections
    // because the caller has indicated they will be writing the data
    // and we do not want the original library cross sections to be changed
    CrossSection2d * writableXs = libraryNuclideXs->getCopy();
    //Check( writableXs != NULL );

    // add the library cross section to the nuclide so
    // writes will be captured in the future
    bool added = nuclide->addNeutron2dData(writableXs);
    //Insist( added, "Was not added." );
    xs = writableXs;
    return xs;
}

/**
    * @brief Obtain the list of mts that are available for the given nuclide
    * @param nuclide - the nuclide for which to obtain the list of mts
    * @param library - the ampx library that backs up the nuclide
    * @return QList<int> * - a newly allocated list of nuclide ids (THIS WILL NEED TO BE DELETED BY YOU)
    */
QList<int> * AmpxDataHelper::getGamma1dXSMts(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);

    QList<int> * mts = new QList<int>();

    //Check( mts != NULL );

    // obtain the mts from the nuclide
    for( int i = 0; i < nuclide->getGamma1dList().size(); i++){
        CrossSection1d * xs1d = nuclide->getGamma1dDataAt(i);
        //Check( xs1d != NULL );
        mts->append(xs1d->getMt());
    }
    int mtCount = mts->size();
    // obtain the mts from the library's nuclide
    if( library == NULL ) return mts; // return if no library was specified
    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return mts; // return if no nuclide is present on the library

        // obtain the mts from the nuclide
    for( int i = 0; i < libraryNuclide->getGamma1dList().size(); i++){
        CrossSection1d * xs1d = libraryNuclide->getGamma1dDataAt(i);
        //Check( xs1d != NULL );
        mts->append(xs1d->getMt());
    }

    // sort in preparation for duplicate mt removal
    qSort(*mts);
    // remove duplicates
    for( int i = mts->size() -1; i >= 1; i-- ){
        if( mts->at(i) == mts->at(i-1) ) mts->removeAt(i);
    }
    qDebug() << "MT total count "<<mts->size()<<" nuclide count "<<mtCount;
    return mts;
}
/**
    * @brief Obtain the neutron cross section 1d with the given mt from either the nuclide or the ampx library
    * This method obtains the cross section 1d with the given mt from the nuclide or the library. The method
    * check the nuclide first, and if the nuclide does not contain the given mt, it attempts to retrieve the cross section
    * from the library.
    *
    * If the cross section is retrieved from the library and readOnly==false, the cross section is copied onto
    * the nuclide, such that any writes will not effect the original library cross section.
    * @param nuclide - the nuclide from which the cross section will come from or be placed on if not present
    * @param library - the library from which the cross section will be obtained if not on the nuclide
    * @param mt - the type of cross section to be obtained
    * @param readOnly - if the cross section is obtained from the library, this indicates if the library needs to be copied onto the nuclide
    * @param CrossSection2d - the desired cross sections obtained from the nuclide or the library, NULL if none were available
    */
CrossSection1d * AmpxDataHelper::getGamma1dXSByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt, bool readOnly ){
    //Require( nuclide != NULL );
    CrossSection1d * xs = NULL;
    // determine if the nuclide already has the desired cross section
    CrossSection1d * nuclideXs = nuclide->getGamma1dDataByMt(mt);
    if( nuclideXs != NULL ){
        qDebug() << "Found neutron1d xs "<<mt<<" on given nuclide";
        xs = nuclideXs; // set the desired cross section
        return xs;
    }

    if( library == NULL ) return xs; // return if no library was specified

    // determine if the library contains the nuclide and the desired cross section
    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());

    if( libraryNuclide == NULL ) return NULL;
    // determine if the library nuclide has the desired cross section
    CrossSection1d * libraryNuclideXs = libraryNuclide->getGamma1dDataByMt(mt);
    if( libraryNuclideXs == NULL ){
        qDebug() << "Failed to find xs by mt "<<mt<<" on either nuclide or library";
        return NULL;
    }

    // if we are read only
    // we have found fulfilled the caller's request
    if( readOnly ){
        qDebug() << "Found neutron1d xs "<<mt<<" on given ampxlibrary";
        xs = libraryNuclideXs;
        return xs;
    }

    // if we are here, than we will need to copy the cross sections
    // because the caller has indicated they will be writing the data
    // and we do not want the original library cross sections to be changed
    CrossSection1d * writableXs = libraryNuclideXs->getCopy();
    //Check( writableXs != NULL );

    // add the library cross section to the nuclide so
    // writes will be captured in the future
    bool added = nuclide->addGamma1dData(writableXs);
    //Insist( added, "Was not added." );
    xs = writableXs;
    qDebug() << "Found neutron1d xs "<<mt<<" on given ampxlibrary and made copy for writing purposes";
    return xs;
}
/**
    * @brief Obtain the list of mts that are available for the given nuclide
    * @param nuclide - the nuclide for which to obtain the list of mts
    * @param library - the ampx library that backs up the nuclide
    * @return QList<int> * - a newly allocated list of nuclide ids (THIS WILL NEED TO BE DELETED BY YOU)
    */
QList<int> * AmpxDataHelper::getGamma2dXSMts(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);

    QList<int> * mts = new QList<int>();

    //Check( mts != NULL );

    // obtain the mts from the nuclide
    for( int i = 0; i < nuclide->getGamma2dList().size(); i++){
        CrossSection2d * xs2d = nuclide->getGamma2dDataAt(i);
        //Check( xs2d != NULL );
        mts->append(xs2d->getMt());
    }

    // obtian the mts from the library's nuclide
    if( library == NULL ) return mts; // return if no library was specified
    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return mts; // return if no nuclide is present on the library

        // obtain the mts from the nuclide
    for( int i = 0; i < libraryNuclide->getGamma2dList().size(); i++){
        CrossSection2d * xs2d = libraryNuclide->getGamma2dDataAt(i);
        //Check( xs2d != NULL );
        mts->append(xs2d->getMt());
    }

    // sort in preparation for duplicate mt removal
    qSort(*mts);
    // remove duplicates
    for( int i = mts->size() -1; i >= 1; i-- ){
        if( mts->at(i) == mts->at(i-1) ) mts->removeAt(i);
    }
    return mts;
}
/**
    * @brief Obtain the gamma cross section 2d with the given mt from either the nuclide or the ampx library
    * This method obtains the cross section 2d with the given mt from the nuclide or the library. The method
    * check the nuclide first, and if the nuclide does not contain the given mt, it attempts to retrieve the cross section
    * from the library.
    *
    * If the cross section is retrieved from the library and readOnly==false, the cross section is copied onto
    * the nuclide, such that any writes will not effect the original library cross section.
    * @param nuclide - the nuclide from which the cross section will come from or be placed on if not present
    * @param library - the library from which the cross section will be obtained if not on the nuclide
    * @param mt - the type of cross section to be obtained
    * @param readOnly - if the cross section is obtained from the library, this indicates if the library needs to be copied onto the nuclide
    * @param CrossSection2d - the desired cross sections obtained from the nuclide or the library, NULL if none were available
    */
CrossSection2d * AmpxDataHelper::getGamma2dXSByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt, bool readOnly){
    //Require( nuclide != NULL );

    CrossSection2d * xs = NULL; // ensure that the xs is NULL

    // determine if the nuclide already has the desired cross section
    CrossSection2d * nuclideXs = nuclide->getGamma2dDataByMt(mt);
    if( nuclideXs != NULL ){
        xs = nuclideXs; // set the desired cross section
        return xs;
    }

    if( library == NULL ) return xs; // return if no library was specified

    // determine if the library contains the nuclide and the desired cross section
    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());

    if( libraryNuclide == NULL ) return NULL;
    // determine if the library nuclide has the desired cross section
    CrossSection2d * libraryNuclideXs = libraryNuclide->getGamma2dDataByMt(mt);
    if( libraryNuclideXs == NULL )  return NULL;

    // if we are read only
    // we have found fulfilled the caller's request
    if( readOnly ){
        xs = libraryNuclideXs;
        return xs;
    }

    // if we are here, than we will need to copy the cross sections
    // because the caller has indicated they will be writing the data
    // and we do not want the original library cross sections to be changed
    CrossSection2d * writableXs = libraryNuclideXs->getCopy();
    //Check( writableXs != NULL );

    // add the library cross section to the nuclide so
    // writes will be captured in the future
    //bool added = nuclide->addGamma2dData(writableXs);
    //Insist( added, "Was not added." );
    xs = writableXs;
    return xs;
}
/**
    * @brief Obtain the list of mts that are available for the given nuclide
    * @param nuclide - the nuclide for which to obtain the list of mts
    * @param library - the ampx library that backs up the nuclide
    * @return QList<int> * - a newly allocated list of nuclide ids (THIS WILL NEED TO BE DELETED BY YOU)
    */
QList<int> * AmpxDataHelper::getGammaProdXSMts(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require(nuclide !=NULL);

    QList<int> * mts = new QList<int>();

    //Check( mts != NULL );

    // obtain the mts from the nuclide
    for( int i = 0; i < nuclide->getGammaProdList().size(); i++){
        CrossSection2d * xs2d = nuclide->getGammaProdDataAt(i);
        //Check( xs2d != NULL );
        mts->append(xs2d->getMt());
    }

    // obtian the mts from the library's nuclide
    if( library == NULL ) return mts; // return if no library was specified
    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());
    if( libraryNuclide == NULL ) return mts; // return if no nuclide is present on the library

        // obtain the mts from the nuclide
    for( int i = 0; i < libraryNuclide->getGammaProdList().size(); i++){
        CrossSection2d * xs2d = libraryNuclide->getGammaProdDataAt(i);
        //Check( xs2d != NULL );
        mts->append(xs2d->getMt());
    }

    // sort in preparation for duplicate mt removal
    qSort(*mts);
    // remove duplicates
    for( int i = mts->size() -1; i >= 1; i-- ){
        if( mts->at(i) == mts->at(i-1) ) mts->removeAt(i);
    }
    return mts;
}
/**
    * @brief Obtain the gamma production cross section 2d with the given mt from either the nuclide or the ampx library
    * This method obtains the cross section 2d with the given mt from the nuclide or the library. The method
    * check the nuclide first, and if the nuclide does not contain the given mt, it attempts to retrieve the cross section
    * from the library.
    *
    * If the cross section is retrieved from the library and readOnly==false, the cross section is copied onto
    * the nuclide, such that any writes will not effect the original library cross section.
    * @param nuclide - the nuclide from which the cross section will come from or be placed on if not present
    * @param library - the library from which the cross section will be obtained if not on the nuclide
    * @param mt - the type of cross section to be obtained
    * @param readOnly - if the cross section is obtained from the library, this indicates if the library needs to be copied onto the nuclide
    * @param CrossSection2d - the desired cross sections obtained from the nuclide or the library, NULL if none were available
    */
CrossSection2d * AmpxDataHelper::getGammaProdXSByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt, bool readOnly){
    //Require( nuclide != NULL );

    CrossSection2d * xs = NULL; // ensure that the xs is NULL

    // determine if the nuclide already has the desired cross section
    CrossSection2d * nuclideXs = nuclide->getGammaProdDataByMt(mt);
    if( nuclideXs != NULL ){
        xs = nuclideXs; // set the desired cross section
        return xs;
    }

    if( library == NULL ) return xs; // return if no library was specified

    // determine if the library contains the nuclide and the desired cross section
    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());

    if( libraryNuclide == NULL ) return NULL;
    // determine if the library nuclide has the desired cross section
    CrossSection2d * libraryNuclideXs = libraryNuclide->getGammaProdDataByMt(mt);
    if( libraryNuclideXs == NULL )  return NULL;

    // if we are read only
    // we have found fulfilled the caller's request
    if( readOnly ){
        xs = libraryNuclideXs;
        return xs;
    }

    // if we are here, than we will need to copy the cross sections
    // because the caller has indicated they will be writing the data
    // and we do not want the original library cross sections to be changed
    CrossSection2d * writableXs = libraryNuclideXs->getCopy();
    //Check( writableXs != NULL );

    // add the library cross section to the nuclide so
    // writes will be captured in the future
    bool added = nuclide->addGammaProdData(writableXs);
    //Insist( added, "Was not added." );
    xs = writableXs;
    return xs;
}
/**
    * @brief Obtain the maximum legendre order for any neutron2d cross section contained in the given nuclide
    * This will be the max legendre order of any neutron2d cross section on the nuclide or the library
    * @param nuclide - the nuclide from which the cross section will come from or be placed on if not present
    * @param library - the library from which the cross section will be obtained if not on the nuclide
    * @return int - the max legendre order for any neutron2d cross section related to the given nuclide
    */
int AmpxDataHelper::getMaxNeutron2dLegendreOrder(LibraryNuclide * nuclide, AmpxLibrary * library ){
    //Require( nuclide != NULL );
    int maxLegendreOrder = 0;

    for( int i = 0; i < nuclide->getNeutron2dList().size(); i++){
        CrossSection2d * xs2d = nuclide->getNeutron2dDataAt(i);
        //Check( xs2d != NULL );
        if( xs2d->getLegendreOrder() > maxLegendreOrder ) maxLegendreOrder = xs2d->getLegendreOrder();
    }

    if( library == NULL ) return maxLegendreOrder;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());

    if( libraryNuclide == NULL ) return maxLegendreOrder;

    for( int i = 0; i < libraryNuclide->getNeutron2dList().size(); i++){
        CrossSection2d * xs2d = libraryNuclide->getNeutron2dDataAt(i);
        //Check( xs2d != NULL );
        if( xs2d->getLegendreOrder() > maxLegendreOrder ) maxLegendreOrder = xs2d->getLegendreOrder();
    }
    return maxLegendreOrder;
}

/**
    * @brief Obtain the maximum legendre order for any gamma2d cross section contained in the given nuclide
    * This will be the max legendre order of any gamma2d cross section on the nuclide or the library
    * @param nuclide - the nuclide from which the cross section will come from or be placed on if not present
    * @param library - the library from which the cross section will be obtained if not on the nuclide
    * @return int - the max legendre order for any gamma2d cross section related to the given nuclide
    */
int AmpxDataHelper::getMaxGamma2dLegendreOrder(LibraryNuclide * nuclide, AmpxLibrary * library ){
    //Require( nuclide != NULL );
    int maxLegendreOrder = 0;

    for( int i = 0; i < nuclide->getGamma2dList().size(); i++){
        CrossSection2d * xs2d = nuclide->getGamma2dDataAt(i);
        //Check( xs2d != NULL );
        if( xs2d->getLegendreOrder() > maxLegendreOrder ) maxLegendreOrder = xs2d->getLegendreOrder();
    }

    if( library == NULL ) return maxLegendreOrder;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());

    if( libraryNuclide == NULL ) return maxLegendreOrder;

    for( int i = 0; i < libraryNuclide->getGamma2dList().size(); i++){
        CrossSection2d * xs2d = libraryNuclide->getGamma2dDataAt(i);
        //Check( xs2d != NULL );
        if( xs2d->getLegendreOrder() > maxLegendreOrder ) maxLegendreOrder = xs2d->getLegendreOrder();
    }
    return maxLegendreOrder;
}

/**
    * @brief Determine if this nuclide contains any Neutron1d data
    * @param nuclide - the nuclide from which to determine the presence of Neutron1d data
    * @param library - the ampx library that backs up this nuclide.
    * @return bool - true, if Neutron1d data is present on either the nuclide or the library nuclide
    *  that backs up the given nuclide
    */
bool AmpxDataHelper::containsNeutron1dData(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require( nuclide != NULL );

    bool dataPresent = nuclide->getNeutron1dList().size() > 0;

    if( dataPresent ) return dataPresent;

    if( library == NULL ) return false;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());

    if( libraryNuclide == NULL ) return false;

    return libraryNuclide->getNeutron1dList().size() > 0;
}
/**
    * @brief Determine if this nuclide contains any Neutron2d data
    * @param nuclide - the nuclide from which to determine the presence of Neutron2d data
    * @param library - the ampx library that backs up this nuclide.
    * @return bool - true, if Neutron2d data is present on either the nuclide or the library nuclide
    *  that backs up the given nuclide
    */
bool AmpxDataHelper::containsNeutron2dData(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require( nuclide != NULL );

    bool dataPresent = nuclide->getNeutron2dList().size() > 0;

    if( dataPresent ) return dataPresent;

    if( library == NULL) return false;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());

    if( libraryNuclide == NULL ) return false;

    return libraryNuclide->getNeutron2dList().size() > 0;
}
/**
    * @brief Determine if this nuclide contains any Gamma1d data
    * @param nuclide - the nuclide from which to determine the presence of Gamma1d data
    * @param library - the ampx library that backs up this nuclide.
    * @return bool - true, if Gamma1d data is present on either the nuclide or the library nuclide
    *  that backs up the given nuclide
    */
bool AmpxDataHelper::containsGamma1dData(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require( nuclide != NULL );

    bool dataPresent = nuclide->getGamma1dList().size() > 0;

    if( dataPresent ) return dataPresent;

    if( library == NULL) return false;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());

    if( libraryNuclide == NULL ) return false;

    return libraryNuclide->getGamma1dList().size() > 0;
}
/**
    * @brief Determine if this nuclide contains any Gamma2d data
    * @param nuclide - the nuclide from which to determine the presence of Gamma2d data
    * @param library - the ampx library that backs up this nuclide.
    * @return bool - true, if Gamma2d data is present on either the nuclide or the library nuclide
    *  that backs up the given nuclide
    */
bool AmpxDataHelper::containsGamma2dData(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require( nuclide != NULL );

    bool dataPresent = nuclide->getGamma2dList().size() > 0;

    if( dataPresent ) return dataPresent;

    if( library == NULL) return false;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());

    if( libraryNuclide == NULL ) return false;

    return libraryNuclide->getGamma2dList().size() > 0;
}

/**
    * @brief Determine if this nuclide contains any GammaProd data
    * @param nuclide - the nuclide from which to determine the presence of GammaProd data
    * @param library - the ampx library that backs up this nuclide.
    * @return bool - true, if GammaProd data is present on either the nuclide or the library nuclide
    *  that backs up the given nuclide
    */
bool AmpxDataHelper::containsGammaProdData(LibraryNuclide * nuclide, AmpxLibrary * library){
    //Require( nuclide != NULL );

    bool dataPresent = nuclide->getGammaProdList().size() > 0;

    if( dataPresent ) return dataPresent;

    if( library == NULL) return false;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());

    if( libraryNuclide == NULL ) return false;

    return libraryNuclide->getGammaProdList().size() > 0;
}

/**
    * @brief Determine if this nuclide contains a given  Neutron1d data
    * @param nuclide - the nuclide from which to determine the presence of Neutron1d data
    * @param library - the ampx library that backs up this nuclide.
    * @return bool - true, if Neutron1d data is present on either the nuclide or the library nuclide
    *  that backs up the given nuclide
    */
bool AmpxDataHelper::containsNeutron1dDataByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt){
    //Require( nuclide != NULL );

    bool dataPresent = nuclide->containsNeutron1dDataByMt(mt);

    if( dataPresent ) return dataPresent;

    if( library == NULL) return false;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());

    if( libraryNuclide == NULL ) return false;

    return libraryNuclide->containsNeutron1dDataByMt(mt);
}
/**
    * @brief Determine if this nuclide contains a given  Neutron2d data
    * @param nuclide - the nuclide from which to determine the presence of Neutron2d data
    * @param library - the ampx library that backs up this nuclide.
    * @return bool - true, if Neutron2d data is present on either the nuclide or the library nuclide
    *  that backs up the given nuclide
    */
bool AmpxDataHelper::containsNeutron2dDataByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt){
    //Require( nuclide != NULL );

    bool dataPresent = nuclide->containsNeutron2dDataByMt(mt);

    if( dataPresent ) return dataPresent;

    if( library == NULL) return false;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());

    if( libraryNuclide == NULL ) return false;

    return libraryNuclide->containsNeutron2dDataByMt(mt);
}
/**
    * @brief Determine if this nuclide contains a given  Gamma1d data
    * @param nuclide - the nuclide from which to determine the presence of Gamma1d data
    * @param library - the ampx library that backs up this nuclide.
    * @return bool - true, if Gamma1d data is present on either the nuclide or the library nuclide
    *  that backs up the given nuclide
    */
bool AmpxDataHelper::containsGamma1dDataByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt){
    //Require( nuclide != NULL );

    bool dataPresent = nuclide->containsGamma1dDataByMt(mt);

    if( dataPresent ) return dataPresent;

    if( library == NULL) return false;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());

    if( libraryNuclide == NULL ) return false;

    return libraryNuclide->containsGamma1dDataByMt(mt);
}
/**
    * @brief Determine if this nuclide contains a given  Gamma2d data
    * @param nuclide - the nuclide from which to determine the presence of Gamma2d data
    * @param library - the ampx library that backs up this nuclide.
    * @return bool - true, if Gamma2d data is present on either the nuclide or the library nuclide
    *  that backs up the given nuclide
    */
bool AmpxDataHelper::containsGamma2dDataByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt){
    //Require( nuclide != NULL );

    bool dataPresent = nuclide->containsGamma2dDataByMt(mt);

    if( dataPresent ) return dataPresent;

    if( library == NULL) return false;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());

    if( libraryNuclide == NULL ) return false;

    return libraryNuclide->containsGamma2dDataByMt(mt);
}

/**
    * @brief Determine if this nuclide contains a given  GammaProd data
    * @param nuclide - the nuclide from which to determine the presence of GammaProd data
    * @param library - the ampx library that backs up this nuclide.
    * @return bool - true, if GammaProd data is present on either the nuclide or the library nuclide
    *  that backs up the given nuclide
    */
bool AmpxDataHelper::containsGammaProdDataByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt){
    //Require( nuclide != NULL );

    bool dataPresent = nuclide->containsGammaProdDataByMt(mt);

    if( dataPresent ) return dataPresent;

    if( library == NULL) return false;

    LibraryNuclide * libraryNuclide = library->getNuclideById(nuclide->getId());

    if( libraryNuclide == NULL ) return false;

    return libraryNuclide->containsGammaProdDataByMt(mt);
}
