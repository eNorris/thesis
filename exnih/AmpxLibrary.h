#ifndef AMPXLIBRARY_H
#define AMPXLIBRARY_H
#include <memory>
#include <iostream>
#include <fstream>
#include <QString>
#include <QList>
#include "Resource.h"
#include "LibraryEnergyBounds.h"
#include "LibraryNuclide.h"
#include "LibraryHeader.h"
#include "CrossSection1d.h"
#include "CrossSection2d.h"
//#include "Standard/Interface/Serializable.h"
using namespace std;


class AmpxLibrary : public Resource{
protected:
   LibraryHeader * header;
   LibraryEnergyBounds * neutronEnergyBounds;
   LibraryEnergyBounds * gammaEnergyBounds;

   QList<LibraryNuclide*> libraryNuclides;

   fstream * file;

   int formatId;

   void initialize();
   /// Remove a nuclide with a given Id from the given list of library nuclides
   /// @param int id - The id of the specified nuclide to remove
   /// @param QList<LibraryNuclide*> list - The list of LibraryNuclides from which to remove
   /// @param bool deleteItem=false - indicate whether to delete the object removed
   /// @return int count - the count of items removed.
   int removeNuclide(int id, QList<LibraryNuclide*>& list, int mixid=0, bool deleteItem=false)
   {
       int count = 0;
       for( int i = 0; i < list.size(); i++){
            if( list.at(i)->getId() == id && list.at(i)->getMixture() == mixid ){
                LibraryNuclide * nuclide = list.value(i);
                list.removeAt(i);
                count++;
                i --;
                if( deleteItem ) delete nuclide;
            }
       }
       return count;
   }// end removeNuclide

public:
   AmpxLibrary();
   AmpxLibrary(const AmpxLibrary & orig);
   virtual ~AmpxLibrary();

   bool operator==(AmpxLibrary & a);

   virtual AmpxLibrary * getCopy() const;

   void setFormatId(int id){
       this->formatId = id;
   }

   int getFormatId(){
       return formatId;
   }

   virtual int open();
   virtual int close();
   bool isOpen();
   QString getTitle() const {return "AmpxLibrary Resource";}
   ///This operation returns the name of the Resource.
   QString getResourceName(){return "AmpxLibrary";}

   LibraryHeader * getLibraryHeader() const { return header;}
   void setLibraryHeader(LibraryHeader * header){
       this->header = header;
   }
   LibraryEnergyBounds * getNeutronEnergyBounds() const { return neutronEnergyBounds;}
   void setNeutronEnergyBounds(LibraryEnergyBounds * bounds){
        this->neutronEnergyBounds = bounds;
   }
   LibraryEnergyBounds * getGammaEnergyBounds() const { return gammaEnergyBounds;}
   void setGammaEnergyBounds(LibraryEnergyBounds * bounds){
        this->gammaEnergyBounds = bounds;
   }
   QList<LibraryNuclide * > getNuclides(){return libraryNuclides;}
   const QList<LibraryNuclide * >& getConstNuclides() const {return libraryNuclides;}
   int getNumberNuclides() const{return libraryNuclides.size();}
   LibraryNuclide * getNuclideAt(int index) const{
       if( index < 0 || index >= libraryNuclides.size()) return NULL;
       return libraryNuclides.value(index);
   }
   /// Obtain a nuclide with the given id
   /// @param int id - The id of the desired nuclide
   /// @param int mixid=0 - The optional mixture id of the desired nuclide.
   /// @return LibraryNuclide * - The desired nuclide, or NULL if no nuclide had the given id
   LibraryNuclide * getNuclideById(int id, int mixid=0) const{
       for( int i = 0; i < libraryNuclides.size(); i ++){
           //std::cout<<" AmpxLibrary.h(88) "<<libraryNuclides[i]->getId() <<" : "<<id <<" : "<< libraryNuclides[i]->getMixture() <<" : "<< mixid<<std::endl;
           if( libraryNuclides[i]->getId() == id && libraryNuclides[i]->getMixture() == mixid)
               return libraryNuclides[i];
       }
       return NULL;
   }

   /// Obtain a list of Nuclides that contain Neutron1d data with the given mt
   /// @param int mt - The reaction type for the given Neutron1d data.
   /// @param int mixid=0 - The optional mixture id of the desired nuclide.
   /// @return QList<LibraryNuclide*>* - A list of all nuclides that contain Neutron1d
   ///                                   cross sections with the desired mt.
   QList<LibraryNuclide*>* getNuclidesWNeutron1dByMt(int mt, int mixid=0){
       QList<LibraryNuclide*>* list = new QList<LibraryNuclide*>();
       for( int i = 0; i < libraryNuclides.size(); i++){
           LibraryNuclide * nuclide = libraryNuclides[i];
           if( nuclide->getMixture() != mixid ) continue;
           if( nuclide->containsNeutron1dDataByMt(mt)) list->append(nuclide);
       }
       if( list->isEmpty() ){
           delete list;
           return NULL;
       }
       return list;
   }
   /// Obtain a list of Nuclides that contain Neutron2d data with the given mt
   /// @param int mt - The reaction type for the given Neutron2d data.
   /// @param int mixid=0 - The optional mixture id of the desired nuclide.
   /// @return QList<LibraryNuclide*>* - A list of all nuclides that contain Neutron2d
   ///                                   cross sections with the desired mt.
   QList<LibraryNuclide*>* getNuclidesWNeutron2dByMt(int mt, int mixid=0){
       QList<LibraryNuclide*>* list = new QList<LibraryNuclide*>();
       for( int i = 0; i < libraryNuclides.size(); i++){
           LibraryNuclide * nuclide = libraryNuclides[i];
           if( nuclide->getMixture() != mixid ) continue;
           if( nuclide->containsNeutron2dDataByMt(mt)) list->append(nuclide);
       }
       if( list->isEmpty() ){
           delete list;
           return NULL;
       }
       return list;
   }
   /// Obtain a list of Nuclides that contain Gamma1d data with the given mt
   /// @param int mt - The reaction type for the given Gamma1d data.
   /// @param int mixid=0 - The optional mixture id of the desired nuclide.
   /// @return QList<LibraryNuclide*>* - A list of all nuclides that contain Gamma1d
   ///                                   cross sections with the desired mt.
   QList<LibraryNuclide*>* getNuclidesWGamma1dByMt(int mt, int mixid=0){
       QList<LibraryNuclide*>* list = new QList<LibraryNuclide*>();
       for( int i = 0; i < libraryNuclides.size(); i++){
           LibraryNuclide * nuclide = libraryNuclides[i];
           if( nuclide->getMixture() != mixid ) continue;
           if( nuclide->containsGamma1dDataByMt(mt)) list->append(nuclide);
       }
       if( list->isEmpty() ){
           delete list;
           return NULL;
       }
       return list;
   }
   /// Obtain a list of Nuclides that contain Gamma2d data with the given mt
   /// @param int mt - The reaction type for the given Gamma2d data.
   /// @param int mixid=0 - The optional mixture id of the desired nuclide.
   /// @return QList<LibraryNuclide*>* - A list of all nuclides that contain Gamma2d
   ///                                   cross sections with the desired mt.
   QList<LibraryNuclide*>* getNuclidesWGamma2dByMt(int mt, int mixid=0){
       QList<LibraryNuclide*>* list = new QList<LibraryNuclide*>();
       for( int i = 0; i < libraryNuclides.size(); i++){
           LibraryNuclide * nuclide = libraryNuclides[i];
           if( nuclide->getMixture() != mixid ) continue;
           if( nuclide->containsGamma2dDataByMt(mt)) list->append(nuclide);
       }
       if( list->isEmpty() ){
           delete list;
           return NULL;
       }
       return list;
   }
   /// Obtain a list of Nuclides that contain GammaProd data with the given mt
   /// @param int mt - The reaction type for the given GammaProd data.
   /// @param int mixid=0 - The optional mixture id of the desired nuclide.
   /// @return QList<LibraryNuclide*>* - A list of all nuclides that contain GammaProd
   ///                                   cross sections with the desired mt.
   QList<LibraryNuclide*>* getNuclidesWGammaProdByMt(int mt, int mixid=0){
       QList<LibraryNuclide*>* list = new QList<LibraryNuclide*>();
       for( int i = 0; i < libraryNuclides.size(); i++){
           LibraryNuclide * nuclide = libraryNuclides[i];
           if( nuclide->getMixture() != mixid ) continue;
           if( nuclide->containsGammaProdDataByMt(mt)) list->append(nuclide);
       }
       if( list->isEmpty() ){
           delete list;
           return NULL;
       }
       return list;
   }
   /// Obtain a list of Nuclides that contain BonData data with the given mt
   /// @param int mt - The reaction type for the given BonData data.
   /// @param int mixid=0 - The optional mixture id of the desired nuclide.
   /// @return QList<LibraryNuclide*>* - A list of all nuclides that contain BonData
   ///                                   with the desired mt.
   QList<LibraryNuclide*>* getNuclidesWBonDataByMt(int mt, int mixid=0){
       QList<LibraryNuclide*>* list = new QList<LibraryNuclide*>();
       for( int i = 0; i < libraryNuclides.size(); i++){
           LibraryNuclide * nuclide = libraryNuclides[i];
           if( nuclide->getMixture() != mixid ) continue;
           if( nuclide->containsBondarenkoDataByMt(mt)) list->append(nuclide);
       }
       if( list->isEmpty() ){
           delete list;
           return NULL;
       }
       return list;
   }
   /// Remove a nuclide with a given Id
   /// This will delete the given nuclide
   /// NOTE: When a nuclide is removed, and deleted,
   /// all references to the nuclide SHOULD NO LONGER BE USED.
   /// @param int id - the id of the nuclide to be removed
   /// @return int - the number of nuclides removed.
   int removeNuclide(int id, int mixid=0){
       int count = removeNuclide(id, libraryNuclides, mixid, true);
              // update the library header's number of nuclides
       if( this->header != NULL ){
           this->header->setNNuc(libraryNuclides.size());
       }
       return count;
   }
   /// Add a nuclide to this AmpxLibrary
   /// The nuclide will be associated with list of nuclides by Mt based on
   /// the current Bondarenko, Neutron and Gamma data and their associated Mt values.
   /// It is important to note that the association will not be built if the
   /// Bondarenkp, Neutron, and/or Gamma data is added after the nuclide is added to
   /// this AmpxLibrary.
   /// @param LibraryNuclide * nuclide - The nuclide to add to this AmpxLibrary
   /// @param bool - true, if the addition was successful, false otherwise.
   bool addNuclide(LibraryNuclide * nuclide){
      // qDebug()<<"Adding nuclide to library!";
       if( nuclide == NULL ) return false;
       libraryNuclides.append(nuclide);

       // update the library header's number of nuclides
       if( this->header != NULL ){
           this->header->setNNuc(libraryNuclides.size());
       }
       return true;
   }
   /// Determine if a nuclide with a given id
   // and optional mixid
   /// is contained in the Ampx Library
   bool containsNuclide(int id, int mixid=0){
       return getNuclideById(id,mixid) != NULL;
   }
   /*!
    * @brief Determine if there are any nuclides with the given material id
    * @param int mix - the mixture for which to determine presence
    * @return bool - true, if nuclides with getMixture()==mix are contained, false otherwise
    */
   bool containsMaterial(int mix){
       for( int i=0; i < libraryNuclides.size(); i ++){
           if( libraryNuclides[i]->getMixture() == mix) return true;
       }
       return false;
   }

     /*!
     * Obtain the maximum order of expansion
     * @return int - the maximum order of expansion
     */
    virtual int getNSCTW();

   /*!
    * @brief Obtain the list of nuclides that are associated with the given material
    * @param int mix - the mixture for which to obtain nuclides
    * @return QList<LibraryNuclide*> * list - the list of nuclides that belong to the given material
    */
   QList<LibraryNuclide*> * getMaterialNuclides(int mix){
       QList<LibraryNuclide*> * list = new QList<LibraryNuclide*>();
       for( int i = 0;i < libraryNuclides.size(); i++){
           LibraryNuclide * nuclide = libraryNuclides[i];
           if( nuclide->getMixture() == mix ) list->append(nuclide);
       }
       if( list->isEmpty() ){
           delete list;
           return NULL;
       }
       return list;
   }
    // serialization interface



public:
    /**
     * @brief the universal version identifier for this class
     */
    static const long uid;
};
typedef std::shared_ptr<AmpxLibrary> SP_AmpxLibrary;

#endif
