//#include "Standard/Interface/AbstractStream.h"
#include <QDebug>
#include "AmpxLibrary.h"
//#include "Resource/AmpxLib/ampxlib_config.h"
#include "LibraryHeader.h"
#include "LibraryNuclide.h"
#include "LibraryEnergyBounds.h"
#include "BondarenkoData.h"
#include "BondarenkoGlobal.h"
#include "NuclideResonance.h"
#include "CrossSection1d.h"
#include "CrossSection2d.h"
#include "resources.h"
#include "AmpxReader.h"
//#include "Standard/Interface/SerialFactory.h"
#include "FileStream.h"
#include "AmpxLibRegistrar.h"
AmpxLibrary::AmpxLibrary()
{
    initialize();
}
AmpxLibrary::AmpxLibrary(const AmpxLibrary & orig)
{
    initialize();
    if( orig.header != NULL ) header = orig.header->getCopy();
    if( orig.neutronEnergyBounds != NULL ) neutronEnergyBounds = orig.neutronEnergyBounds->getCopy();
    if( orig.gammaEnergyBounds != NULL ) gammaEnergyBounds = orig.gammaEnergyBounds->getCopy();

    for( int i = 0; i < orig.libraryNuclides.size(); i++){
        libraryNuclides.append(orig.libraryNuclides[i]->getCopy());
    }

    formatId = orig.formatId;
}
AmpxLibrary * AmpxLibrary::getCopy() const {
    return new AmpxLibrary(*this);
}
AmpxLibrary::~AmpxLibrary()
{
    if( isOpen() ) close();
    if( header != NULL ) delete header;
    if( neutronEnergyBounds != NULL ) delete neutronEnergyBounds;
    if( gammaEnergyBounds != NULL ) delete gammaEnergyBounds;
    if( file != NULL ) delete file;
    for( int i = 0; i < libraryNuclides.size(); i++)
        delete libraryNuclides.value(i);
    libraryNuclides.clear();
}

bool AmpxLibrary::operator==(AmpxLibrary & a){
    if( header != NULL && a.header != NULL ){
        bool equal = *header == *a.header;
        if( !equal ){
            qDebug() << qPrintable("AmpxHeaders not equal!");
            return false;
        }
    }else if( header != a.header){
        qDebug() << qPrintable("AmpxHeaders not equal!");
        return false;
    }
    if( neutronEnergyBounds != NULL && a.neutronEnergyBounds != NULL ){
        bool equal = *neutronEnergyBounds == *a.neutronEnergyBounds;
        if( !equal ){
            qDebug() << qPrintable("Ampx Neutron bounds not equal!");
            return false;
        }
    }else if( neutronEnergyBounds != a.neutronEnergyBounds){
        qDebug() << qPrintable("Ampx Neutron bounds not equal!");
        return false;
    }
    if( gammaEnergyBounds != NULL && a.gammaEnergyBounds != NULL ){
        bool equal = *gammaEnergyBounds == *a.gammaEnergyBounds;
        if( !equal ){
            qDebug() << qPrintable("Ampx Gamma bounds not equal!");
            return false;
        }
    }else if( gammaEnergyBounds != a.gammaEnergyBounds){
        qDebug() << qPrintable("Ampx Gamma bounds not equal!");
        return false;
    }

    for( int i = 0; i < libraryNuclides.size(); i++){
        bool equal = *libraryNuclides[i] == *a.libraryNuclides[i];
        if( !equal ){
            qDebug() << qPrintable("Ampx Nuclide at inddex ") << (i+1) << " not equal!";
            return false;
        }
    }
    qDebug() << qPrintable("Ampx Libraries are identical according to equality operator!");
    return true;
}
void AmpxLibrary::initialize()
{
    header = NULL;
    neutronEnergyBounds = NULL;
    gammaEnergyBounds = NULL;
    file = NULL;
    formatId = -1;
}

int AmpxLibrary::open()
{
    if( getFileName() == "" ) return -1;
    if( isOpen() ) return -2;

    file = new fstream(qPrintable(getFileName()), ios::in | ios::binary );

    AmpxReader reader;
    int rtncode = reader.read(file, *this);
    // attempt to deserialize a serialized library if legacy failed
    if( rtncode != 0 ) {
        //Standard::SerialFactory objectFactory;
        //AmpxLibRegistrar( &objectFactory );
        FileStream stream(file);
        stream.setReadHead(0); // reading required to start from start of file
        //stream.setFactory(&objectFactory);
        //if( stream.isNext(AmpxLibrary::uid) ){
        //    int deserializeCode = this->deserialize(&stream);
        //    if( deserializeCode != 0 ) rtncode = deserializeCode * 100;
        //}else{
        //    qDebug() << qPrintable("***Error: Failed to open library "<<getFileName().toStdString()<<std::endl
        //                     <<reader.getReadError().toStdString());
        //}
    }
    return rtncode;
}// end of open

int AmpxLibrary::close()
{
    if( !isOpen() ) return -1;
    if( file != NULL) {
       file->close();
       delete file;
       file = NULL;
    }
    formatId = -1;
    return 0;
}// end of close

bool AmpxLibrary::isOpen()
{
    if( file != NULL){    // user intended open via direct call to open
        return file->is_open();
    }
    if( formatId != -1) return true;  // was read by the loader
    return false;
}// end of isOpen

// serializable interface
const long AmpxLibrary::uid = 0x961323556f1bda6a;




int AmpxLibrary::getNSCTW(){
    int maxOrderExpansion = 0;
    int l;
    for( int i = 0; i < this->getNumberNuclides(); i++){
       LibraryNuclide * nuc = this->getNuclideAt(i);
       CrossSection2d  *xs = nuc->getTotal2dXS();
       if( xs != 0) {
           l = xs->getLegendreOrder();
           if( maxOrderExpansion < l) maxOrderExpansion = l;
       }
       for( int ii = 0; ii < nuc->getNumNeutron2dData(); ii++){
           xs = nuc->getNeutron2dDataAt(ii);
           l = xs->getLegendreOrder();
            if( maxOrderExpansion < l) maxOrderExpansion = l;
       }
       for( int ii = 0; ii < nuc->getNumGamma2dData(); ii++){
           xs = nuc->getGamma2dDataAt(ii);
           l = xs->getLegendreOrder();
            if( maxOrderExpansion < l) maxOrderExpansion = l;
       }
       for( int ii = 0; ii < nuc->getNumGammaProdData(); ii++){
           xs = nuc->getGammaProdDataAt(ii);
           l = xs->getLegendreOrder();
           if( maxOrderExpansion < l) maxOrderExpansion = l;
       }
    }
    return (maxOrderExpansion+1)/2;
}

