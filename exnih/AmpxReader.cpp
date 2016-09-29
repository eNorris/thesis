#include <stdio.h>
#include <string.h>
#include <istream>
#include <QDebug>
#include "AmpxReader.h"
#include "LibraryHeader.h"
#include "LibraryEnergyBounds.h"
#include "LibraryNuclide.h"
#include "NuclideResonance.h"
#include "BondarenkoGlobal.h"
#include "BondarenkoData.h"
#include "BondarenkoInfiniteDiluted.h"
#include "BondarenkoFactors.h"
#include "CrossSection1d.h"
#include "CrossSection2d.h"
#include "ScatterMatrix.h"
#include "resources.h"
#include "EndianUtils.h"
//#include "Resource/AmpxLib/ampxlib_config.h"

//using namespace ScaleUtils::IO;

AmpxReader::AmpxReader()
{
    initialize();
}

void AmpxReader::initialize()
{
    swapEndian = false;
}

int AmpxReader::footprintLibraryEnergyBounds(int numGrps)
{
    return 2*(numGrps+1)*sizeof(float);
}

int AmpxReader::footprintBondarenkoGlobal(int numSig0, int numTemp)
{
    return ((numSig0+numTemp)*sizeof(float))
           + (sizeof(float)*2); /// elo ehi
}// end of footprint

int AmpxReader::footprintLibraryNuclide(){
    return AMPX_NUCLIDE_DESCR_LENGTH+
         (sizeof(int)*AMPX_NUMBER_NUCLIDE_INTEGER_OPTIONS);
}

int AmpxReader::footprintBondarenkoInfiniteDiluted(int startGrp, int endGrp){
        return ((endGrp-startGrp + 1)*sizeof(float));
}// end of footprint

int AmpxReader::footprintBondarenkoFactors(int numSig, int numTemp, int startGrp, int endGrp){
    return (numSig*numTemp*(endGrp-startGrp + 1)*sizeof(float));
}// end of footprint

int AmpxReader::footprintCrossSection1d(int numGrps){
        return ((numGrps + 1)*sizeof(float));
}// end of footprint

int AmpxReader::footprintCrossSection2d(){
    return CROSS_SECTION_2D_INTEGER_OPTIONS * sizeof(int);
}

int AmpxReader::footprintNuclideResonance(int numResolved, int numUnresolved){
    return ((numResolved+numUnresolved)*sizeof(float));
}// end of footprint

int AmpxReader::footprintScatterMatrix(int length, int type){
    int size = 0;
    switch(type){
        default:
        case AMPXLIB_NEUTRON2D_DATA:
        case AMPXLIB_GAMMA2D_DATA:
            size=(length*sizeof(float)); /// length of data
            break;
        case AMPXLIB_GAMMAPRODUCTION_DATA:
            /// gamma production uses self defining format
            ///'L,X(I),I=1,L'
            size=((length+1)*sizeof(float)); /// length of data
            break;
    }
    return size;
}// end of footprint
int AmpxReader::footprintLibraryHeader()
{
    return AMPX_HEADER_TITLE_LENGTH + (sizeof(int)*AMPX_HEADER_NUMBER_INTEGER_OPTIONS);
}
LibraryHeader * AmpxReader::readLibraryHeader(fstream & is,
                       int * pos,
                       bool verbose)
{
    if( !is.is_open() ){
       if( verbose ) cout<<"File stream is NULL, LibraryHeader read failed!"<<endl;
       return NULL;
    }
    int previousPos = is.tellg();
    if( pos != NULL ) is.seekg(*pos); /// seek to desired read position
    LibraryHeader * header = new LibraryHeader();

    /// fortran record header
    int libraryRecordSize = footprintLibraryHeader();
    int headerSize;
    is.read((char*)&headerSize, sizeof(int));

    /// determine endianess
    if( headerSize != libraryRecordSize ){
        if( EndianUtils::reverse_int_bytes(headerSize) == libraryRecordSize ){
            swapEndian = true;
        }
        else{
            if( this->printErrors ){
                cerr<<"***Error: AmpxReader cannot determine Library's endianess."
                    <<endl
                    <<"Please ensure you are attempting to load an AmpxLibrary."
                    <<endl;
            }
            this->readError += "***Error: AmpxReader cannot determine Library's endianess.\n";
            this->readError += "Please ensure you are attempting to load an AmpxLibrary.\n";
            delete header;
            return NULL;
        }
    }
    int members[AMPX_HEADER_NUMBER_INTEGER_OPTIONS];
    /// read from disk, a library header
    is.read((char*)members, AMPX_HEADER_NUMBER_INTEGER_OPTIONS*sizeof(int));
    header->setIdTape(members[0]);
    header->setNNuc(members[1]);
    header->setIGM(members[2]);
    header->setIFTG(members[3]);
    header->setMSN(members[4]);
    header->setIPM(members[5]);
    header->setI1(members[6]);
    header->setI2(members[7]);
    header->setI3(members[8]);
    header->setI4(members[9]);
    is.read((char*)header->getITM(), AMPX_HEADER_TITLE_LENGTH);
    if( swapEndian ) /// big endian, need to convert
    {
        headerSize = EndianUtils::reverse_int_bytes(headerSize);
        /// convert big endian to native
        header->setIdTape(EndianUtils::reverse_int_bytes(header->getIdTape()));
        header->setNNuc(EndianUtils::reverse_int_bytes(header->getNNuc()));
        header->setIGM(EndianUtils::reverse_int_bytes(header->getIGM()));
        header->setIFTG(EndianUtils::reverse_int_bytes(header->getIFTG()));
        header->setMSN(EndianUtils::reverse_int_bytes(header->getMSN()));
        header->setIPM(EndianUtils::reverse_int_bytes(header->getIPM()));
        header->setI1(EndianUtils::reverse_int_bytes(header->getI1()));
        header->setI2(EndianUtils::reverse_int_bytes(header->getI2()));
        header->setI3(EndianUtils::reverse_int_bytes(header->getI3()));
        header->setI4(EndianUtils::reverse_int_bytes(header->getI4()));
    }
    /// null terminate
    header->getITM()[AMPX_HEADER_TITLE_LENGTH]='\0';
    /// fortran record footer
    int footerSize;
    is.read((char*)&footerSize, sizeof(int));
    /// ensure sizes match
    if( libraryRecordSize != headerSize ){
        delete header;
        if( this->printErrors ){
            cerr<<"LibraryHeader byte count does not match on-disk LibraryHeader byte count!"<<endl;
        }
        this->readError += "LibraryHeader byte count does not match on-disk LibraryHeader byte count!\n";
        is.seekg(previousPos); /// make sure to reset read head
        return NULL;
    }
    if( verbose ) cout<<qPrintable(header->toQString())<<endl;
    //qDebug()<<header->toQString();
    return header;
}
LibraryEnergyBounds * AmpxReader::readEnergyBounds(fstream & is,
                                    int numGroups,
                                    int * pos,
                                    bool verbose)
{
    if( !is.is_open() ){
        if( this->printErrors ){
            cerr<<"***Error: Attempting to read energy bounds from file that is not open."<<endl;
        }
        this->readError += "***Error: Attempting to read energy bounds from file that is not open.\n";
        return NULL;
    }
    if( numGroups <= 0 ) return NULL;
    int previousPos = is.tellg();
    if( pos != NULL ) is.seekg(*pos);

    LibraryEnergyBounds * bounds = new LibraryEnergyBounds();
    bounds->resizeBounds(numGroups+1);
    bounds->resizeLethargy(numGroups+1);

    int boundsSize = footprintLibraryEnergyBounds(numGroups);
    int headerSize;
    is.read((char*)&headerSize, sizeof(int));
    is.read((char*)bounds->getBounds(), sizeof(float)*(numGroups+1));
    is.read((char*)bounds->getLethargyBounds(), sizeof(float)*(numGroups+1));
    if( swapEndian ) /// dealing with endian
    {
        headerSize = EndianUtils::reverse_int_bytes(headerSize);
        for( int i=0; i < numGroups+1; i++ ){
           bounds->getBounds()[i]=EndianUtils::reverse_float_bytes(bounds->getBounds()[i]);
           bounds->getLethargyBounds()[i]=EndianUtils::reverse_float_bytes(bounds->getLethargyBounds()[i]);
        }
    }

    // account for possibly weighted data being appended after bounds
    if( headerSize > boundsSize ){
        this->readError += "***Warning: Energy Bounds record size mismatch occurred... possibly weighted data?\n";
        this->readError += QString("---expected %1, found %2---\n").arg(boundsSize).arg(headerSize);
        int size2Skip = headerSize - boundsSize;
        is.ignore(size2Skip);
        boundsSize = headerSize;
    }
    int footerSize;
    is.read((char*)&footerSize, sizeof(int));
    if( headerSize != boundsSize ){
        if( this->printErrors ){
            cerr<<"***Error: While attempting to read Energy bounds record size mismatch occured!"<<endl;
            cerr<<"---expected "<<boundsSize<<", found "<<headerSize<<"---"<<endl;
        }
        this->readError += "***Error: While attempting to read Energy bounds record size mismatch occured!\n";
        this->readError += QString("---expected %1, found %2---\n").arg(boundsSize).arg(headerSize);
        delete bounds;
        is.seekg(previousPos);
        return NULL;
    }
    return bounds;
}
LibraryNuclide* AmpxReader::readLibraryNuclide(fstream & is,
                        int * pos,
                        bool verbose)
{
   if(!is.is_open() ) return NULL;

   if( pos != NULL ) is.seekg(*pos);

   int previousPos = is.tellg();
   int recordSize = footprintLibraryNuclide();
   int header=0;
   is.read((char*)&header, sizeof(int));
   if(verbose) cout<<"Reading nuclide at pos="<<is.tellg()<<endl;
   LibraryNuclide * nuclide = new LibraryNuclide();

   is.read((char*)nuclide->getDescription(), AMPX_NUCLIDE_DESCR_LENGTH);
   is.read((char*)nuclide->getWords(), sizeof(int)*AMPX_NUMBER_NUCLIDE_INTEGER_OPTIONS);
   nuclide->getDescription()[AMPX_NUCLIDE_DESCR_LENGTH] = '\0';
   /// check endian and convert if needed
   if( swapEndian ){
       header = EndianUtils::reverse_int_bytes(header);

       for( int i = 0; i < AMPX_NUMBER_NUCLIDE_INTEGER_OPTIONS; i++)
           nuclide->getWords()[i] = EndianUtils::reverse_int_bytes(nuclide->getWords()[i]);
   }
   if( recordSize != header){
       delete nuclide;
       if(this->printErrors ){
           cerr<<"***Error: Library Nuclide record size does not match on disk record size"<<endl;
           cerr<<"---expected "<<recordSize;
           cerr<<", found "<<header<<"---"<<endl;
       }
       this->readError += "***Error: Library Nuclide record size does not match on disk record size\n";
       this->readError += QString("---expected %1, found %2---\n").arg(recordSize).arg(header);
       is.seekg(previousPos);
       return NULL;
   }
   int footer;
   is.read((char*)&footer, sizeof(int));

   if( swapEndian ) footer = EndianUtils::reverse_int_bytes(footer);
   if( footer != header){
       delete nuclide;
       if( this->printErrors ){
           cerr<<"***Error: Library Nuclide record header and footer do not match!"<<endl;
           cerr<<"---header "<<header<<", footer "<<footer<<"---"<<endl;
       }
       this->readError += "***Error: Library Nuclide record header and footer do not match!\n";
       this->readError += QString("---header %1, footer %2---\n").arg(header).arg(footer);
       is.seekg(previousPos);
       return NULL;
   }
   if( verbose ) cout<<qPrintable(nuclide->toQString());
   //qDebug()<<nuclide->toQString();
   return nuclide;
}
BondarenkoGlobal * AmpxReader::readBondarenkoGlobal(fstream & is,
                                    int numSig0,
                                    int numTemps,
                                    int * pos,
                                    bool verbose)
{
    if( !is.is_open() ) return NULL;
    int previousPos = is.tellg();
    if( pos != NULL ) is.seekg(*pos);
    if( numSig0 <= 0 && numTemps <= 0 ){

        return NULL;
    }
    int recordSize = footprintBondarenkoGlobal(numSig0, numTemps);
    BondarenkoGlobal* bondGlobal = new BondarenkoGlobal();
    int header;
    is.read((char*)&header, sizeof(int));
    if( numSig0 > 0 ){
        float * sig0 = new float[numSig0];
        is.read((char*)sig0, sizeof(float)*numSig0);
        bondGlobal->setSig0(sig0, numSig0);
    }
    if( numTemps > 0 ){
        float * temps = new float[numTemps];
        is.read((char*)temps, sizeof(float)*numTemps);
        bondGlobal->setTemps(temps, numTemps);
    }

    float eTmp;
    is.read((char*)&eTmp, sizeof(float));
    bondGlobal->setElo(eTmp);
    is.read((char*)&eTmp, sizeof(float));
    bondGlobal->setEhi(eTmp);

    if( swapEndian ) // endian switch
    {
        header = EndianUtils::reverse_int_bytes(header);
        bondGlobal->setElo(EndianUtils::reverse_float_bytes(bondGlobal->getElo()));
        bondGlobal->setEhi(EndianUtils::reverse_float_bytes(bondGlobal->getEhi()));
        for( int i = 0; i < numSig0; i++)
        {
            bondGlobal->getSig0()[i] = EndianUtils::reverse_float_bytes(bondGlobal->getSig0()[i]);
        }
        for( int i = 0; i < numTemps; i++)
        {
            bondGlobal->getTemps()[i] = EndianUtils::reverse_float_bytes(bondGlobal->getTemps()[i]);
        }
    }
    if( header != recordSize ){
        if( this->printErrors ){
            cerr<<"***Error: Bondarenko Global object record size does not match that on disk!"<<endl;
            cerr<<"---expected "<<recordSize<<", found "<<header<<"---"<<endl;
        }
        this->readError += "***Error: Bondarenko Global object record size does not match that on disk!\n";
        this->readError += QString("---expected %1, found %2---\n").arg(recordSize).arg(header);
        delete bondGlobal;
        is.seekg(previousPos);
        return NULL;
    }
    int footer;
    is.read((char*)&footer, sizeof(int));

    if( verbose ) cout<<qPrintable(bondGlobal->toQString());
    //qDebug()<<bondGlobal->toQString();
    return bondGlobal;
}
int AmpxReader::footprintBondarenkoData()
{
    return sizeof(int) * AMPX_BONDARENKO_DATA_INTEGER_OPTIONS;
}
QList<BondarenkoData*> AmpxReader::readBondarenkoData(fstream & is,
                int numBondSets,
                int numBondSig0,
                int numBondTemps,
                int * pos, bool verbose)
{
    QList<BondarenkoData*> set;
    if( !is.is_open() ) return set;
    if( numBondSets <=0 ) return set;
    int previousPos = is.tellg();
    if( pos != NULL ) is.seekg(*pos);
    int *mt =  new int[numBondSets];
    int *nf = new int[numBondSets];
    int *nl = new int[numBondSets];
    int *order = new int[numBondSets];
    int *ioff = new int[numBondSets];
    int *nz = new int[numBondSets];

    int recordSize = numBondSets*footprintBondarenkoData();
    int header;
    is.read((char*)&header,sizeof(int));
    is.read((char*)mt,sizeof(int)*numBondSets);
    is.read((char*)nf,sizeof(int)*numBondSets);
    is.read((char*)nl,sizeof(int)*numBondSets);
    is.read((char*)order,sizeof(int)*numBondSets);
    is.read((char*)ioff,sizeof(int)*numBondSets);
    is.read((char*)nz,sizeof(int)*numBondSets);
    if( swapEndian ) /// big endian, switch
    {
        header = EndianUtils::reverse_int_bytes(header);
        for( int i =0; i < numBondSets; i++){
            mt[i] = EndianUtils::reverse_int_bytes(mt[i]);
            nf[i] = EndianUtils::reverse_int_bytes(nf[i]);
            nl[i] = EndianUtils::reverse_int_bytes(nl[i]);
            order[i] = EndianUtils::reverse_int_bytes(order[i]);
            ioff[i] = EndianUtils::reverse_int_bytes(ioff[i]);
            nz[i] = EndianUtils::reverse_int_bytes(nz[i]);
        }
    }
    int footer;
    is.read((char*)&footer,sizeof(int));
    /// check the record sizes to ensure proper record
    if( header != recordSize ){
        if( this->printErrors ){
            cerr<<"***Error: Record size for BondarenkoData does not match object size!"<<endl;
            cerr<<"---expected "<<recordSize<<", found "<<header<<"---"<<endl;
        }
        this->readError += "***Error: Record size for BondarenkoData does not match object size!\n";
        this->readError += QString("---expected %1, found %2---\n").arg(recordSize).arg(header);
        is.seekg(previousPos);
                delete [] mt;
                delete [] nf;
                delete [] nl;
                delete [] order;
                delete [] ioff;
                delete [] nz;
        return set;
    }

    /// loop num bondsets and create their objects
    for( int i = 0; i < numBondSets; i++)
    {
        BondarenkoData * bond = new BondarenkoData();
        bond->setData(AMPX_BONDARENKO_MT,mt[i]);
        bond->setData(AMPX_BONDARENKO_NF,nf[i]);
        bond->setData(AMPX_BONDARENKO_NL,nl[i]);
        bond->setData(AMPX_BONDARENKO_ORDER,order[i]);
        bond->setData(AMPX_BONDARENKO_IOFF,ioff[i]);
        bond->setData(AMPX_BONDARENKO_NZ,nz[i]);

        set.append(bond);
        if( verbose ) cout<<qPrintable(bond->toQString())<<endl;
        //qDebug()<<bond->toQString();
        BondarenkoInfiniteDiluted * bondInfDil = readBondarenkoInfiniteDiluted(is,
                                                 bond->getData(AMPX_BONDARENKO_NF),
                                                 bond->getData(AMPX_BONDARENKO_NL),
                                                 NULL, verbose);
       if( bondInfDil == NULL ){
                        delete [] mt;
                        delete [] nf;
                        delete [] nl;
                        delete [] order;
                        delete [] ioff;
                        delete [] nz;
           return set;
       }
       bond->setInfiniteDiluted(bondInfDil);
       BondarenkoFactors * factors = readBondarenkoFactors(is,
                                                 numBondSig0, numBondTemps,
                                                 bond->getData(AMPX_BONDARENKO_NF),
                                                 bond->getData(AMPX_BONDARENKO_NL),
                                                 NULL, verbose);
           if( factors == NULL ){
                   delete [] mt;
                   delete [] nf;
                   delete [] nl;
                   delete [] order;
                   delete [] ioff;
                   delete [] nz;
                   return set;
           }
       bond->setFactors(factors);
    }
    delete [] mt;
    delete [] nf;
    delete [] nl;
    delete [] order;
    delete [] ioff;
    delete [] nz;
    return set;
}
BondarenkoInfiniteDiluted * AmpxReader::readBondarenkoInfiniteDiluted(fstream & is,
                                                    int startGrp,
                                                    int endGrp,
                                                    int * pos,
                                                    bool verbose)
{
    if( !is.is_open() ) return NULL;
    int previousPos = is.tellg();
    if( pos != NULL ) is.seekg(*pos);
    int recordSize = footprintBondarenkoInfiniteDiluted(startGrp, endGrp);
    BondarenkoInfiniteDiluted* bondInfiniteDiluted = new BondarenkoInfiniteDiluted();
    int header;
    is.read((char*)&header, sizeof(int));
    int numGrps = endGrp - startGrp + 1;
    if( numGrps > 0 ){
        float * values = new float[numGrps];
        is.read((char*)values, sizeof(float)*numGrps);
        bondInfiniteDiluted->setValues(values, numGrps);
    }
    else{
        delete bondInfiniteDiluted;
        is.seekg(previousPos);
        if( this->printErrors ){
            cerr<<"***Error: BondarenkoInfiniteDiluted object cannot cover negative group range!"<<endl;
        }
        this->readError +="***Error: BondarenkoInfiniteDiluted object cannot cover negative group range!\n";
        return NULL;
    }

    if( swapEndian ) // endian switch
    {
        header = EndianUtils::reverse_int_bytes(header);
        for( int i = 0; i < numGrps; i++)
        {
            bondInfiniteDiluted->getValues()[i] = EndianUtils::reverse_float_bytes(bondInfiniteDiluted->getValues()[i]);
        }
    }
    if( header != recordSize ){
        if( this->printErrors ){
            cerr<<"***Error: Bondarenko InfiniteDiluted object record size does not match that on disk!"<<endl;
            cerr<<"---expected "<<recordSize<<", found "<<header<<"---"<<endl;
        }
        this->readError += "***Error: Bondarenko InfiniteDiluted object record size does not match that on disk!\n";
        this->readError += QString("---expected %1, found %2---\n").arg(recordSize).arg(header);
        delete bondInfiniteDiluted;
        return NULL;
    }
    int footer;
    is.read((char*)&footer, sizeof(int));

    if( verbose ) cout<<qPrintable(bondInfiniteDiluted->toQString());
    //qDebug()<<bondInfiniteDiluted->toQString();
    return bondInfiniteDiluted;
}
BondarenkoFactors * AmpxReader::readBondarenkoFactors(fstream & is,
                                        int numSig0,
                                        int numTemps,
                                        int startGrp,
                                        int endGrp,
                                        int * pos,
                                        bool verbose)
{
    if( !is.is_open() ) return NULL;
    int previousPos = is.tellg();
    if( pos != NULL ) is.seekg(*pos);
    int recordSize = footprintBondarenkoFactors(numSig0, numTemps,startGrp, endGrp);
    BondarenkoFactors* bondFactors = new BondarenkoFactors();
    int header;
    is.read((char*)&header, sizeof(int));
    int factorsLength = numSig0*numTemps*(endGrp - startGrp + 1);
    if( factorsLength > 0 ){
        float * values = new float[factorsLength];
        is.read((char*)values, sizeof(float)*factorsLength);
        bondFactors->setValues(values, numSig0, numTemps, (endGrp-startGrp+1));
    }
    else{
        delete bondFactors;
        is.seekg(previousPos);
        if( this->printErrors ){
            cerr<<"***Error: BondarenkoFactors object cannot cover negative group range!"<<endl;
        }
        this->readError += "***Error: BondarenkoFactors object cannot cover negative group range!\n";
        return NULL;
    }

    if( swapEndian ) // endian switch
    {
        header = EndianUtils::reverse_int_bytes(header);
        for( int i = 0; i < factorsLength; i++)
        {
            bondFactors->getValues()[i] = EndianUtils::reverse_float_bytes(bondFactors->getValues()[i]);
        }
    }
    if( header != recordSize ){
        if( this->printErrors ){
            cerr<<"***Error: Bondarenko Factors object record size does not match that on disk!"<<endl;
            cerr<<"---expected "<<recordSize<<", found "<<header<<"---"<<endl;
        }
        this->readError += "***Error: Bondarenko Factors object record size does not match that on disk!\n";
        this->readError += QString("---expected %1, found %2---\n").arg(recordSize).arg(header);
        delete bondFactors;
        return NULL;
    }
    int footer;
    is.read((char*)&footer, sizeof(int));
    if( verbose ) cout<<qPrintable(bondFactors->toQString());
    //qDebug()<<bondFactors->toQString();
    return bondFactors;
}
QList<CrossSection1d *> AmpxReader::readCrossSection1d(fstream & is,
                                        int num1d,
                                        int numGrps,
                                        int * pos,
                                        bool verbose)
{
    QList<CrossSection1d *> set;
    if( !is.is_open() ) return set;
    if( pos != NULL ) is.seekg(*pos);


    int recordSize = num1d * footprintCrossSection1d(numGrps);
    int header;
    is.read((char*)&header, sizeof(int));
    if( numGrps <= 0 || num1d <= 0 ){
        if( this->printErrors ){
            cerr<<"***Error: Attempting to read CrossSection1d data where either the number of 1d, or number of groups are zero!"<<endl;
        }
        this->readError += "***Error: Attempting to read CrossSection1d data where either the number of 1d, or number of groups are zero!\n";
        return set;
    }
    if( swapEndian )
        header = EndianUtils::reverse_int_bytes(header);
    for( int j = 0; j < num1d; j++ ){
        float mt;
        is.read((char*)&mt, sizeof(float));
        float * values = new float[numGrps];
        is.read((char*)values, sizeof(float)*numGrps);
        CrossSection1d * crossSection1d = new CrossSection1d();
        crossSection1d->setValues(values, numGrps);
        crossSection1d->setMt((int)mt);
        if( swapEndian ) // endian switch
        {
            crossSection1d->setMt((int)EndianUtils::reverse_float_bytes(mt));
            for( int i = 0; i < numGrps; i++)
            {
                crossSection1d->getValues()[i] = EndianUtils::reverse_float_bytes(crossSection1d->getValues()[i]);
            }
        }

        set.append(crossSection1d);
        if( verbose ) cout<<qPrintable(crossSection1d->toQString());
        //qDebug()<<crossSection1d->toQString();
    }
    if( header != recordSize ){
        if( this->printErrors ){
            cerr<<"***Error: CrossSection1d  object record size does not match that on disk!"<<endl;
            cerr<<"---expected "<<recordSize<<", found "<<header<<"---"<<endl;
        }
        this->readError += "***Error: CrossSection1d  object record size does not match that on disk!\n";
        this->readError += QString("---expected %1, found %2---\n").arg(recordSize).arg(header);
        return set;
    }
    int footer;
    is.read((char*)&footer, sizeof(int));

    return set;
}
QList<CrossSection2d*> AmpxReader::readCrossSection2d(fstream & is,
                                        int num2d,
                                        int type,
                                        int * pos,
                                        bool verbose)
{
    QList<CrossSection2d *> set;
    if( !is.is_open() ) return set;
    if( pos != NULL ) is.seekg(*pos);

    int recordSize = num2d * footprintCrossSection2d();
    int header;
    is.read((char*)&header, sizeof(int));
    if( num2d <= 0 ){
        if( this->printErrors ){
            cerr<<"***Error: Attempting to read CrossSection2d data where the number of 2d is zero!"<<endl;
        }
        this->readError += "***Error: Attempting to read CrossSection2d data where the number of 2d is zero!\n";
        return set;
    }
    int *mt = new int[num2d];
    int * length = new int[num2d];
    int * nl = new int[num2d];
    int * nt = new int[num2d];
    is.read((char*)mt, num2d*sizeof(int));
    is.read((char*)length, num2d*sizeof(int));
    is.read((char*)nl, num2d*sizeof(int));
    is.read((char*)nt, num2d*sizeof(int));
    int footer;
    is.read((char*)&footer, sizeof(int));
    if( header != footer ){
        if( this->printErrors ){
            cerr<<"***Error: Attempting to read CrossSection2d, Header ("
                <<header<<") and Footer ("
                <<footer<<") do not match!"<<endl;
        }
        this->readError += QString("***Error: Attempting to read CrossSection2d, Header (%1) and Footer (%2) do not match!\n").arg(header).arg(footer);
                delete [] mt;
                delete [] length;
                delete [] nl;
                delete [] nt;
        return set;
    }
    if( swapEndian )
        header = EndianUtils::reverse_int_bytes(header);
    for( int j = 0; j < num2d; j++ ){
        CrossSection2d * crossSection2d = new CrossSection2d();
        if( swapEndian ) // endian switch
        {
            mt[j] = EndianUtils::reverse_int_bytes(mt[j]);
            length[j] = EndianUtils::reverse_int_bytes(length[j]);
            nl[j] = EndianUtils::reverse_int_bytes(nl[j]);
            nt[j] = EndianUtils::reverse_int_bytes(nt[j]);
        }
        crossSection2d->setData(0,mt[j]);
        crossSection2d->setData(1,length[j]);
        crossSection2d->setData(2,nl[j]);
        crossSection2d->setData(3,nt[j]);

        crossSection2d->setType(type);
        set.append(crossSection2d);
        if( verbose ) cout<<qPrintable(crossSection2d->toQString())<<endl;
        //qDebug()<<crossSection2d->toQString();
    }
    if( header != recordSize ){
        if( this->printErrors ){
            cerr<<"***Error: CrossSection2d  object record size does not match that on disk!"<<endl;
            cerr<<"---expected "<<recordSize<<", found "<<header<<"---"<<endl;
        }
        this->readError += "***Error: CrossSection2d  object record size does not match that on disk!\n";
        this->readError += QString("---expected %1, found %2---\n").arg(recordSize).arg(header);
        delete [] mt;
        delete [] length;
        delete [] nl;
        delete [] nt;
        return set;
    }

    for(int i = 0; i < num2d; i++){
        QList<ScatterMatrix*> matrices = readScatterMatrices(is,
                                 set.at(i)->getData(CROSS_SECTION_2D_NT),
                                 set.at(i)->getData(CROSS_SECTION_2D_NL),
                                 set.at(i)->getData(CROSS_SECTION_2D_LENGTH),
                                 type,
                                 NULL, verbose);
        set.at(i)->setScatterMatrices(matrices);
    }
    delete [] mt;
    delete [] length;
    delete [] nl;
    delete [] nt;
    return set;
}
ScatterMatrix * AmpxReader::readScatterMatrix(fstream & is,
                                int length,
                                int type,
                                int * pos,
                                bool verbose)
{
    if( !is.is_open() ) return NULL;
    int previousPos = is.tellg();
    if( pos != NULL ) is.seekg(*pos);
    int recordSize = footprintScatterMatrix(length,type);
    if( length <= 0 ){
        if( this->printErrors ){
            cerr<<"***Error: Attempting to read ScatterMatrix data where length is zero!"<<endl;
        }
        this->readError += "***Error: Attempting to read ScatterMatrix data where length is zero!\n";
        return NULL;
    }
    int header;
    is.read((char*)&header, sizeof(int));
    if( swapEndian )
        header = EndianUtils::reverse_int_bytes(header);
    if( type == AMPXLIB_GAMMAPRODUCTION_DATA )
    {
        is.read((char*)&recordSize,sizeof(int));
        if( swapEndian )
            recordSize = EndianUtils::reverse_int_bytes(recordSize);
        recordSize *= sizeof(float);
        header-=sizeof(float); /// adjust for reduced recordSize
    }
    length = recordSize/sizeof(float);
    float * data = new float[length];

    is.read((char*)data, recordSize);
    ScatterMatrix * scatterMatrix = new ScatterMatrix();
    if( swapEndian ) // endian switch
    {
        for( int i = 0; i < length; i++)
        {
            data[i] = EndianUtils::reverse_float_bytes(data[i]);
        }
    }
    if( verbose )cout<<"Expanding sink data for scatter matrix legendreOrder="<<scatterMatrix->getLegendreOrder()
                     <<", temp="<<scatterMatrix->getTemp()<<endl;
    QList<SinkGroup*> sinkGroups = SinkGroup::expandMagicWordArray(data,length, verbose);
    scatterMatrix->setSinkGroups(sinkGroups);

    if( verbose ) cout<<qPrintable(scatterMatrix->toQString());
    //qDebug()<<scatterMatrix->toQString();

    if( header != recordSize ){
        if( this->printErrors ){
            cerr<<"***Error: ScatterMatrix  object record size does not match that on disk!"<<endl;
            cerr<<"---expected "<<recordSize<<", found "<<header<<"---"<<endl;
        }
        this->readError += "***Error: ScatterMatrix  object record size does not match that on disk!\n";
        this->readError += QString("---expected %1, found %2---\n").arg(recordSize).arg(header);
        is.seekg(previousPos);
        delete scatterMatrix;
        delete [] data;
        return NULL;
    }
    int footer;
    is.read((char*)&footer, sizeof(int));
    if( swapEndian) footer = EndianUtils::reverse_int_bytes(footer);
    if( type == AMPXLIB_GAMMAPRODUCTION_DATA ) footer -= sizeof(float);
    if( header != footer ){
        if( this->printErrors ){
            cerr<<"***Error: Attempting to read ScatterMatrix, header and footer do not match!"<<endl;
            cerr<<"---Header "<<header<<" --- Footer "<<footer<<" ---"<<endl;
        }
        this->readError += "***Error: Attempting to read ScatterMatrix, header and footer do not match!\n";
        this->readError += QString("---Header %1 --- Footer %2 ---\n").arg(header).arg(footer);
        delete scatterMatrix;
        delete [] data;
        is.seekg(previousPos);
        return NULL;
    }
    delete [] data;
    return scatterMatrix;
}
QList<ScatterMatrix *> AmpxReader::readScatterMatrices(fstream & is,
                                int numTemps,
                                int maxLegendreOrder,
                                int length,
                                int type,
                                int * pos,
                                bool verbose)
{
    QList<ScatterMatrix *> set;
    if( !is.is_open() ) return set;
    int previousPos = is.tellg();
    if( pos != NULL ) is.seekg(*pos);

    int maxTemp = (numTemps > 1 ? numTemps : 1);
    int maxLegendre = maxLegendreOrder+1;
    float * temperatureData = new float[maxTemp];
    /// read temperature from disk
    if( numTemps >= 1 && type != AMPXLIB_GAMMAPRODUCTION_DATA){
        int tempHeader;
        is.read((char*)&tempHeader, sizeof(int));
        is.read((char*)temperatureData,sizeof(float)*numTemps);
        int tempfooter;
        is.read((char*)&tempfooter, sizeof(int));
        if( tempHeader != tempfooter ) {
            if( this->printErrors ){
                cerr<<"***Error: Attempting to read ScatterMatrix temp, Header and Footer do not match!"<<endl
                    <<"--- header "<<tempHeader<<", footer "<<tempfooter<<"---"<<endl;
            }
            this->readError +="***Error: Attempting to read ScatterMatrix temp, Header and Footer do not match!\n";
            this->readError += QString("--- header %1, footer %2---\n").arg(tempHeader).arg(tempfooter);
            is.seekg(previousPos);
                        delete [] temperatureData;
            return set;
        }
        if( swapEndian )
            tempHeader = EndianUtils::reverse_int_bytes(tempHeader);
        for( int i = 0; i < numTemps; i ++){
            if( swapEndian )
                temperatureData[i] = EndianUtils::reverse_float_bytes(temperatureData[i]);
            if( verbose ) cout<<"Temp["<<i<<"]="<<temperatureData[i]<<endl;
        }
        if( tempHeader != (int)sizeof(float)*numTemps )
        {
            if( this->printErrors ){
                cerr<<"***Error: Attempting to read ScatterMatrix temperature data, record size does not match!"<<endl;
                cerr<<"---expected "<<(sizeof(float)*numTemps)<<", found "<<tempHeader<<"---"<<endl;
            }
            this->readError += "***Error: Attempting to read ScatterMatrix temperature data, record size does not match!\n";
            this->readError += QString("---expected %1, found %2---\n").arg(sizeof(float)*numTemps).arg(tempHeader);
            delete [] temperatureData;
            is.seekg(previousPos);
            return set;
        }
    }else{
        if( verbose ){
            cout<<"Setting temperature to 0.0"<<endl;
        }
        temperatureData[0] = 0.0;
    }

    for( int t = 0; t < maxTemp; t++){
        for( int l = 0; l < maxLegendre; l++ ){
            ScatterMatrix * scatterMatrix = readScatterMatrix(is,length,type, NULL, verbose);
            if( scatterMatrix == NULL ){
                if( this->printErrors ){
                    cerr<<"***Failed to read scatter matrix!"<<endl;
                }
                this->readError += "***Failed to read scatter matrix!\n";
                delete [] temperatureData;
                is.seekg(previousPos);
                return set;
            }
            scatterMatrix->setTemp(temperatureData[t]);
            scatterMatrix->setLegendreOrder(l);
            //qDebug()<<scatterMatrix->toQString();
            set.append(scatterMatrix);
        }
    }
    delete [] temperatureData;
    return set;
}
NuclideResonance * AmpxReader::readNuclideResonance(fstream & is,
                                    int numResolved,
                                    int numUnresolved,
                                    int * pos,
                                    bool verbose)
{
    if( !is.is_open() ) return NULL;
    int previousPos = is.tellg();
    if( pos != NULL ) is.seekg(*pos);
    // this hack fixes the endf5 files that contain nuclideResonance
    // hack provided by Doro
    numResolved *= 6;
    numUnresolved += 9;
    int recordSize = footprintNuclideResonance(numResolved, numUnresolved);
    NuclideResonance* resonance = new NuclideResonance();
    int header;
    is.read((char*)&header, sizeof(int));
    if( numResolved > 0 ){
        float * resolved = new float[numResolved];
        is.read((char*)resolved, sizeof(float)*(numResolved));
        resonance->setResolved(resolved, numResolved);
    }
    if( numResolved > 0 ){
        float * unresolved = new float[numUnresolved];
        is.read((char*)unresolved, sizeof(float)*(numUnresolved));
        resonance->setUnresolved(unresolved, numUnresolved);
    }

    if( swapEndian ) // endian switch
    {
        header = EndianUtils::reverse_int_bytes(header);
        for( int i = 0; i < numResolved; i++)
        {
            resonance->getResolved()[i] = EndianUtils::reverse_float_bytes(resonance->getResolved()[i]);
        }
        for( int i = 0; i < numUnresolved; i++)
        {
            resonance->getUnresolved()[i] = EndianUtils::reverse_float_bytes(resonance->getUnresolved()[i]);
        }
    }
    if( header != recordSize ){
        if( this->printErrors ){
            cerr<<"***Error: NuclideResonance object record size does not match that on disk!"<<endl;
            cerr<<"---expected "<<recordSize<<", found "<<header<<"---"<<endl;
        }
        this->readError+="***Error: NuclideResonance object record size does not match that on disk!\n";
        this->readError+=QString("---expected %1, found %2---\n").arg(recordSize).arg(header);
        delete resonance;
        is.seekg(previousPos);
        return NULL;
    }
    int footer;
    is.read((char*)&footer, sizeof(int));
    if( swapEndian ){
        footer = EndianUtils::reverse_int_bytes(footer);
    }
    if( header != footer ){
        if(this->printErrors){
            cerr<<"***Error: Attempting to read NuclideResonance, Header ("
            <<header<<") and Footer ("
            <<footer<<") do not match!"<<endl;
        }
        this->readError+=QString("***Error: Attempting to read NuclideResonance, Header (%1) and Footer (%2) do not match!\n").arg(header).arg(footer);
        delete resonance;
        is.seekg(previousPos);
        return NULL;
    }

    if( verbose ) cout<<qPrintable(resonance->toQString());
    return resonance;
}

bool AmpxReader::readLibraryHeader(fstream & file, AmpxLibrary & library)
{
    /// obtain library header
    library.setLibraryHeader(readLibraryHeader(file, NULL, this->verboseOutput));
    if( library.getLibraryHeader() == NULL ) return false;
    return true;
}
bool AmpxReader::readNuclideInfo(fstream & file, AmpxLibrary & library)
{
    if( library.getLibraryHeader() == NULL ) return false;

    int numberNuclideDataSets = library.getLibraryHeader()->getNNuc();
    /// loop for number of nuclide data sets attempting to read
    ///for( int i = 0; i < numberNuclideDataSets; i++ )
    ///{
    ///    LibraryNuclide * nuclide = LibraryNuclide::read(file, NULL, true);
    ///    if( nuclide == NULL ) return false;
    ///    libraryNuclides.append(nuclide);
    ///} /// end of loop over nuclide data set

    /// ignore the nuclide listings, these will be duplicated after energy bounds
    streamsize size2Ignore = numberNuclideDataSets *
                     (headerSize()+footerSize() + footprintLibraryNuclide());

    file.ignore(size2Ignore);
    return true;
} // end of readNuclideInfo


bool AmpxReader::readEnergyBounds(fstream &file,AmpxLibrary & library)
{
    if( library.getLibraryHeader() == NULL ) return false;

    /// obtain library energy bounds
    int numNeutronGroups, numGammaGroups;
    numNeutronGroups = library.getLibraryHeader()->getIGM();
    numGammaGroups   = library.getLibraryHeader()->getIPM();

    library.setNeutronEnergyBounds(readEnergyBounds(file, numNeutronGroups, NULL, this->verboseOutput));
    if( numNeutronGroups != 0 && library.getNeutronEnergyBounds() == NULL ) return false;

    library.setGammaEnergyBounds(readEnergyBounds(file, numGammaGroups, NULL, this->verboseOutput));
    if( numGammaGroups != 0 && library.getGammaEnergyBounds() == NULL ) return false;

    return true;
}// end of readEnergyBounds
bool AmpxReader::readNuclideData(fstream &file, AmpxLibrary & library)
{
    bool verbose = this->verboseOutput;
    if( library.getLibraryHeader() == NULL ) return false;
    int numNuclides = library.getLibraryHeader()->getNNuc();

    for( int i = 0; i < numNuclides; i++ )
    {
        std::cout << "Reading nuclide " << i << std::endl;
        LibraryNuclide * nuclide = readLibraryNuclide(file, NULL, verbose);
        if( nuclide == NULL ){
            if(this->printErrors)cerr<<"***Error: unable to read the nuclide at index "<<i+1<<"!"<<endl;
            this->readError += QString("***Error: unable to read the nuclide at index %1!").arg(i+1);
            return false;
        }
        QString nuclideDescription = QString("nuclide %1 at index %2").arg(nuclide->getId()).arg(i+1);
        if( nuclide->getNumBondarenkoSig0Data() > 0 ||
                               nuclide->getNumBondarenkoTempData()>0){
            /// read bondarenko global data (sig0, temps, elo, ehi)
            BondarenkoGlobal * bondGlobal  = readBondarenkoGlobal(file,
                               nuclide->getNumBondarenkoSig0Data(),
                               nuclide->getNumBondarenkoTempData(), NULL, verbose);
            if( bondGlobal == NULL )
            {
                if(this->printErrors)cerr<<"***Error: "<<qPrintable(nuclideDescription)<<" global bondarenko data failed to be read!"<<endl;
                this->readError+=QString("***Error: %1 nuclide's global bondarenko data failed to be read!\n").arg(nuclideDescription);
                delete nuclide;
                return false;
            }
            nuclide->setBondarenkoGlobal(bondGlobal);
        }
        /// read/log bondarenko data
        QList<BondarenkoData *> bondData = readBondarenkoData(file,
                                    nuclide->getNumBondarenkoData(),
                                    nuclide->getNumBondarenkoSig0Data(),
                                    nuclide->getNumBondarenkoTempData(), NULL, verbose);
        nuclide->setBondarenkoList(bondData);
        /// resonance data
        if( nuclide->getNumResolvedResonance()!= 0 ||
                                     nuclide->getNumEnergiesUnresolvedResonance()!=0 )
        {
            NuclideResonance * resonance = readNuclideResonance(file,
                                     nuclide->getNumResolvedResonance(),
                                     nuclide->getNumEnergiesUnresolvedResonance(), NULL, verbose);
            if( resonance == NULL ){
                if(this->printErrors)cerr<<"***Error: "<<qPrintable(nuclideDescription)<<" resonance data failed to be read!"<<endl;
                this->readError+=QString("***Error: %1 resonance data failed to be read!\n").arg(nuclideDescription);
                delete nuclide;
                return false;
            }
            nuclide->setResonance(resonance);
        }

        /// neutron 1d
        if(nuclide->getNumNeutron1dData() > 0 && library.getLibraryHeader()->getIGM() > 0 )
        {
            //qDebug()<<"Reading Neutron1d CrossSections...";
            QList<CrossSection1d*> neutron1d = readCrossSection1d(file,
                                         nuclide->getNumNeutron1dData(),
                                         library.getLibraryHeader()->getIGM(),NULL, verbose);
            if( neutron1d.size() != nuclide->getNumNeutron1dData() ){
                if( this->printErrors ){
                    cerr<<"***Warning: "<<qPrintable(nuclideDescription)<<" number of neutron1d cross-sections is not what was expected!"<<endl;
                    cerr<<"---expected "<<nuclide->getNumNeutron1dData()<<", read "<<neutron1d.size()<<"---"<<endl;
                }
                this->readError += QString("***Warning: %1 number of neutron1d cross-sections is not what was expected!\n").arg(nuclideDescription);
                this->readError += QString("---expected %1, read %2---\n").arg(nuclide->getNumNeutron1dData()).arg(neutron1d.size());
            }
            nuclide->setNeutron1dList(neutron1d);

        }
        /// neutron 2d
        if(nuclide->getNumNeutron2dData() > 0 )
        {
            //qDebug()<<"Reading Neutron2d CrossSections...";
            QList<CrossSection2d*> neutron2d = readCrossSection2d(file,
                                         nuclide->getNumNeutron2dData(),
                                         AMPXLIB_NEUTRON2D_DATA,NULL, verbose);
            if( neutron2d.size() != nuclide->getNumNeutron2dData() ){
                if( this->printErrors ){
                    cerr<<"***Warning: "<<qPrintable(nuclideDescription)<<" number of neutron2d cross-sections is not what was expected!"<<endl;
                    cerr<<"---expected "<<nuclide->getNumNeutron2dData()<<", read "<<neutron2d.size()<<"---"<<endl;
                }
                this->readError += QString("***Warning: %1 number of neutron2d cross-sections is not what was expected!\n").arg(nuclideDescription);
                this->readError += QString("---expected %1, read %2---\n").arg(nuclide->getNumNeutron2dData()).arg(neutron2d.size());
            }
            nuclide->setNeutron2dList(neutron2d);

        }
        /// gamma production (neutron2gamma)
        if(nuclide->getNumGammaProdData() > 0 )
        {
            //qDebug()<<"Reading GammaProduction CrossSections...";
            QList<CrossSection2d*> neutron2Gam2d = readCrossSection2d(file,
                                         nuclide->getNumGammaProdData(),
                                         AMPXLIB_GAMMAPRODUCTION_DATA, NULL, verbose);
            if( neutron2Gam2d.size() != nuclide->getNumGammaProdData() ){
                if( this->printErrors ){
                    cerr<<"***Warning: "<<qPrintable(nuclideDescription)<<" number of gamma production cross-sections is not what was expected!"<<endl;
                    cerr<<"---expected "<<nuclide->getNumGammaProdData()<<", read "<<neutron2Gam2d.size()<<"---"<<endl;
                }
                this->readError += QString("***Warning: %1 number of gamma production cross-sections is not what was expected!\n").arg(nuclideDescription);
                this->readError += QString("---expected %1, read %2---\n").arg(nuclide->getNumGammaProdData()).arg(neutron2Gam2d.size());
            }
            nuclide->setGammaProdList(neutron2Gam2d);

        }
        /// gamma 1d
        if(nuclide->getNumGamma1dData()> 0 && library.getLibraryHeader()->getIPM() > 0 ){
            //qDebug()<<"Reading Gamma1d CrossSections...";
            QList<CrossSection1d*> gamma1d = readCrossSection1d(file,
                                         nuclide->getNumGamma1dData(),
                                         library.getLibraryHeader()->getIPM(), NULL, verbose);
            if( gamma1d.size() != nuclide->getNumGamma1dData() ){
                if( this->printErrors ){
                    cerr<<"***Warning: "<<qPrintable(nuclideDescription)<<" number of gamma1d cross-sections is not what was expected!"<<endl;
                    cerr<<"---expected "<<nuclide->getNumGamma1dData()<<", read "<<gamma1d.size()<<"---"<<endl;
                }
                this->readError += QString("***Warning: %1 number of gamma1d cross-sections is not what was expected!\n").arg(nuclideDescription);
                this->readError += QString("---expected %1, read %2---\n").arg(nuclide->getNumGamma1dData()).arg(gamma1d.size());
            }
            nuclide->setGamma1dList(gamma1d);
        }
        /// gamma 2d
        if(nuclide->getNumGamma2dData() > 0 )
        {
            //qDebug()<<"Reading Gamma2d CrossSections...";
            QList<CrossSection2d*> gamma2d = readCrossSection2d(file,
                                         nuclide->getNumGamma2dData(),
                                         AMPXLIB_GAMMA2D_DATA, NULL, verbose);
            if( gamma2d.size() != nuclide->getNumGamma2dData() ){
                if( this->printErrors ){
                    cerr<<"***Warning: "<<qPrintable(nuclideDescription)<<" number of gamma2d cross-sections is not what was expected!"<<endl;
                    cerr<<"---expected "<<nuclide->getNumGamma2dData()<<", read "<<gamma2d.size()<<"---"<<endl;
                }
                this->readError += QString("***Warning: %1 number of gamma2d cross-sections is not what was expected!\n").arg(nuclideDescription);
                this->readError += QString("---expected %1, read %2---\n").arg(nuclide->getNumGamma2dData()).arg(gamma2d.size());
            }
            nuclide->setGamma2dList(gamma2d);
        }
        library.addNuclide(nuclide);
    } // end of loop over nuclide data set
    return true;
}// end of readNuclideData

int AmpxReader::read(fstream * file, AmpxLibrary & library, bool printErrors, bool verbose)
{
    this->printErrors = printErrors;
    this->verboseOutput = verbose;
    /// try to read LibraryHeader
    if( !readLibraryHeader( *file, library ) ) return -3;
    /// try to read the Nuclide header information
    if( !readNuclideInfo(*file, library) ) return -4;
    /// try to read the library energy bounds information
    if( !readEnergyBounds(*file, library) ) return -5;

    /// try to read the nuclide info and their associated
    /// bondarenko, neutron1d,2d, gamma1d, 2d, etc...
    if( !readNuclideData(*file, library) ) return -6;
    return 0;
}


int AmpxReader::readHeaderInfo(fstream * file, AmpxLibrary & library, bool printErrors, bool verbose){
    this->printErrors = printErrors;
    this->verboseOutput = verbose;

    if( !readLibraryHeader( *file, library ) ) return -3;
    /// try to read the Nuclide header information
    if( !readNuclideInfo(*file, library) ) return -4;
    /// try to read the library energy bounds information
    if( !readEnergyBounds(*file, library) ) return -5;
    return 0;
}
