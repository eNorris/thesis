#ifndef LIBRARYNUCLIDE_H
#define LIBRARYNUCLIDE_H

#include <QAtomicInt>
#include <QString>
#include <QList>
#include <QDebug>
#include <string.h>
//#include "Standard/Interface/Serializable.h"

#include "LibraryItem.h"
#include "resources.h"
#include "BondarenkoData.h"
#include "BondarenkoGlobal.h"
#include "CrossSection2d.h"
#include "CrossSection1d.h"
#include "NuclideFilter.h"
#include "SubGroupData.h"
#include "LibrarySourceDefs.h"

using namespace std;

#define AMPX_NUCLIDE_INTEGER_OPTION_START 19
#define AMPX_NUMBER_NUCLIDE_INTEGER_OPTIONS  32
#define AMPX_NUCLIDE_DESCR_LENGTH (4*18)
#define AMPX_MAST_NUCLIDE_ID   0
#define AMPX_NUCLIDE_WORD_19   0
#define AMPX_MAST_NUCLIDE_NUM_RESOLVED_RESDATA   1
#define AMPX_WORK_NUCLIDE_SET_DERIVED_FROM 1
#define AMPX_NUCLIDE_WORD_20   1
#define AMPX_MAST_NUCLIDE_NUM_ENERGIES_UNRESOLVED_RESDATA   2
#define AMPX_WORK_NUCLIDE_ZONE_NUMBER 2
#define AMPX_NUCLIDE_WORD_21   2
#define AMPX_MAST_NUCLIDE_NUM_NEU_1D_DATA   3
#define AMPX_WORK_NUCLIDE_NUM_ZONES_FROM_PRODUCER_PROBLEM 3
#define AMPX_NUCLIDE_WORD_22   3
#define AMPX_MAST_NUCLIDE_NUM_NEU_2D_DATA   4
#define AMPX_WORK_NUCLIDE_P0_TOTAL_LENGTH 4
#define AMPX_NUCLIDE_WORD_23   4
#define AMPX_WORK_NUCLIDE_TOTAL_ORDER_OF_EXPANSION 5
#define AMPX_NUCLIDE_WORD_24   5
#define AMPX_MAST_NUCLIDE_NUM_GAM_1D_DATA   6
#define AMPX_WORK_NUCLIDE_ZONE_WEIGHTED_SEQUENCE_SET 6
#define AMPX_NUCLIDE_WORD_25   6
#define AMPX_MAST_NUCLIDE_NUM_GAM_2D_DATA   7
#define AMPX_WORK_NUCLIDE_NUM_ZONE_WEIGHTED_SETS 7
#define AMPX_NUCLIDE_WORD_26   7
#define AMPX_MAST_NUCLIDE_NUM_NEU2GAM_2D_DATA   8
#define AMPX_WORK_NUCLIDE_MAX_LENGTH_SCATTER_ARRAY 8
#define AMPX_NUCLIDE_WORD_27   8
#define AMPX_WORK_NUCLIDE_NUM_NEU_1D_DATA 9
#define AMPX_NUCLIDE_WORD_28   9
#define AMPX_MAST_NUCLIDE_A 10
#define AMPX_NUCLIDE_WORD_29  10
#define AMPX_MAST_NUCLIDE_ZA 11
#define AMPX_NUCLIDE_WORD_30  11
#define AMPX_NUCLIDE_WORD_31  12
#define AMPX_NUCLIDE_MIXTURE 13
#define AMPX_NUCLIDE_WORD_32  13
#define AMPX_NUCLIDE_WORD_33  14
#define AMPX_MAST_NUCLIDE_POWER_PER_FISSION  15
#define AMPX_NUCLIDE_WORD_34  15
#define AMPX_MAST_NUCLIDE_ENERGY_RELEASE_PER_CAP  16
#define AMPX_NUCLIDE_WORD_35  16
#define AMPX_MAST_NUCLIDE_MAX_LENGTH_SCATTER_MATRIX  17
#define AMPX_NUCLIDE_WORD_36  17
#define AMPX_MAST_NUCLIDE_NUM_BOND_SETS 18
#define AMPX_NUCLIDE_WORD_37  18
#define AMPX_MAST_NUCLIDE_NUM_BOND_SIG_SETS 19
#define AMPX_NUCLIDE_WORD_38  19
#define AMPX_MAST_NUCLIDE_NUM_BOND_TEMP_SETS 20
#define AMPX_NUCLIDE_WORD_39  20
#define AMPX_MAST_NUCLIDE_MAX_NUM_BOND_GRPS 21
#define AMPX_NUCLIDE_WORD_40  21
#define AMPX_WORK_NUCLIDE_NUM_GAM_1D_DATA 22
#define AMPX_NUCLIDE_WORD_41  22
#define AMPX_NUCLIDE_WORD_42  23
#define AMPX_MAST_NUCLIDE_POTEN_SCATTER  24
#define AMPX_NUCLIDE_WORD_43  24
#define AMPX_NUCLIDE_WORD_44  25
#define AMPX_MAST_NUCLIDE_FAST_NEU_DATA 26
#define AMPX_NUCLIDE_WORD_45  26
#define AMPX_MAST_NUCLIDE_THERM_NEU_DATA 27
#define AMPX_NUCLIDE_WORD_46  27
#define AMPX_MAST_NUCLIDE_GAM_DATA 28
#define AMPX_NUCLIDE_WORD_47  28
#define AMPX_MAST_NUCLIDE_GAMPROD_DATA 29
#define AMPX_NUCLIDE_WORD_48  29
#define AMPX_NUCLIDE_WORD_49  30
#define AMPX_NUCLIDE_SOURCE 30
#define AMPX_MAST_NUCLIDE_NUM_RECORD_SETS  31
#define AMPX_NUCLIDE_WORD_50  31
#define AMPX_MAST_SAMPLE_NO   5

class NuclideResonance;

class LibraryNuclideData{
public:
    QAtomicInt ref;
    char description[AMPX_NUCLIDE_DESCR_LENGTH+1]; /// description
    int words[AMPX_NUMBER_NUCLIDE_INTEGER_OPTIONS];
    LibraryNuclideData(): ref(1){
        strcpy(description, "");
        for( int i = 0; i < AMPX_NUMBER_NUCLIDE_INTEGER_OPTIONS; i++)
            words[i] = 0;
    }
    LibraryNuclideData(const LibraryNuclideData & orig): ref(1){
        for( int i = 0; i < AMPX_NUCLIDE_DESCR_LENGTH+1; i++)
            description[i] = orig.description[i];
        for( int i = 0; i < AMPX_NUMBER_NUCLIDE_INTEGER_OPTIONS; i++)
            words[i] = orig.words[i];
    }
};

class LibraryNuclide : public LibraryItem { //, public Standard::Serializable{
private:
  LibraryNuclideData * d;

  QList<BondarenkoData*> bondarenkoData;
  /// global sigma, temp and elo/ehi data
  BondarenkoGlobal * bondarenkoGlobalData;
  SubGroupData * subGroups;
  NuclideResonance * resonance;
  QList<CrossSection1d*> neutron1dData;
  QList<CrossSection1d*> gamma1dData;

  /// cross section 2d
  QList<CrossSection2d*> neutron2dData;
  QList<CrossSection2d*> gamma2dData;
  QList<CrossSection2d*> gammaProdData;

  CrossSection2d * totalXS;
  void initialize();
public:

public:
  LibraryNuclide();
  LibraryNuclide( const LibraryNuclide & orig, NuclideFilter * filter=NULL);
  virtual ~LibraryNuclide();

  virtual bool operator==(LibraryNuclide & a);

  /*! Obtain a shallow copy of this LibraryNuclide
  * @return LibraryNuclide * - The shallow copy of the LibraryNuclide
  * */
  virtual LibraryNuclide * getCopy() const;
  /*! Obtain a filtered copy of this LibraryNuclide
  * @return LibraryNuclide * - The shallow copy of the LibraryNuclide
  * */
  virtual LibraryNuclide * getCopy(NuclideFilter& filter);

  /*! Populate this LibraryNuclide with a copy of the given nuclide
  * @param LibraryNuclide & nuclide - the nuclide from which to copy
   * */
  virtual void copyFrom(LibraryNuclide & nuclide, bool fixAsMaster=false);

  /*! Obtain the array of words associated with this nuclide
  * This is primary a convenience method.
   * */
  int * getWords(){return d->words;}

  /*! Set the total cross section for this nuclide
   @param CrossSection2d * total - the total 2d cross section.*/
  void setTotal2dXS(CrossSection2d * total){this->totalXS = total;}

  /*! Obtain the total scattering matrices for this nuclide.
   **/
  CrossSection2d * getTotal2dXS(){return this->totalXS;}
  void removeTotal2dXS(){
      if( this->totalXS != NULL ){
          delete this->totalXS;
          this->totalXS = NULL;
      }
  }
  /*! Determine if total cross section is available
   * @return bool - true if available, false otherwise */
  bool containsTotal2dXS(){return this->totalXS != NULL;}

  bool containsBondarenkoDataByMt(int mt);
  bool containsNeutron1dDataByMt(int mt);
  bool containsNeutron2dDataByMt(int mt);
  bool containsGamma1dDataByMt(int mt);
  bool containsGamma2dDataByMt(int mt);
  bool containsGammaProdDataByMt(int mt);

  /*! Obtain the bondarenko global data
  * that contains the global sigma and temperature values for all
  * bondarenko factors associated with this nuclide. It also contains
  * the upper and lower energy for which factors can apply
  * @return BondarenkoGlobal * - The global bondarenko data for this nuclide
   * */
  BondarenkoGlobal * getBondarenkoGlobal(){return bondarenkoGlobalData;}

  /*! set the global bondarenko data object for this nuclide */
  void setBondarenkoGlobal(BondarenkoGlobal * global){

      bondarenkoGlobalData = global;
      if( global != NULL ){
          this->setNumBondarenkoSig0Data(global->getSig0Size());
          this->setNumBondarenkoTempData(global->getTempsSize());
      }else{
          this->setNumBondarenkoSig0Data(0);
          this->setNumBondarenkoTempData(0);
      }
  }

  bool hasSubGroupData(){
      if( subGroups != 0) return true;
      return false;
  }
  SubGroupData * getSubGroupData(){ return subGroups;}
  void setSubGroupData(SubGroupData * grps){
      if( subGroups != 0) delete subGroups;
      subGroups = grps;
  }


  /*! obtain the internal list of BondarenkoData */
  QList<BondarenkoData*> getBondarenkoList(){return bondarenkoData;}

  /*! Obtain the bondarenko data object at a given index
  * @return BondarenkoData * - The object requested, null if index is out of bounds*/
  BondarenkoData* getBondarenkoDataAt(int index)
  {
      if( index < 0 || index >= bondarenkoData.size()) return NULL;
      return bondarenkoData.value(index);
  }
  /*! Remove all BondarenkoData with a specified MT
  * @param int mt - the mt for which to remove by
  * @return int - The number of BondarenkoData removed
   * */
  int removeBondarenkoData(int mt)
  {
       int count = 0;
       for( int i =0; i < bondarenkoData.size(); i ++){
       BondarenkoData * ptr = bondarenkoData.value(i);
           if( ptr->getMt() == mt ){
                   bondarenkoData.removeAt(i);
                   i--;
                   count++;
                   delete ptr;
           }
       }

       this->setNumBondarenkoData(bondarenkoData.size());
       return count;
  }
  void removeAllBondarenkoData(){
      if( bondarenkoGlobalData != NULL ){
          delete bondarenkoGlobalData;
          bondarenkoGlobalData = NULL;
      }
      for( int i = 0 ; i < bondarenkoData.size(); i++ ){
          delete bondarenkoData[i];
      }
      bondarenkoData.clear();
      this->setNumBondarenkoSig0Data(0);
      this->setNumBondarenkoTempData(0);
      this->setNumBondarenkoData(0);
      this->setMaxNumBondGrp(0);
  }

  void removeSubGroupData(){
      this->setSubGroupData(0);
  }

  /*! Add a BondarenkoData object to this LibraryNuclide
  * If a BondarenkoData object with the same Mt already exists as
  * part of this LibraryNuclide, it will be removed (deleted) and replaced.
  * @param BondarenkoData * ptr - The pointer to the object to add
  * @return bool - true if BondarenkoData was succesfully added, false otherwise.
   * */
  bool addBondarenkoData(BondarenkoData * ptr)
  {
      if( ptr == NULL ) return false;
      bondarenkoData.append(ptr);
      this->setNumBondarenkoData(bondarenkoData.size());
      return true;
  }
  /*! set the list of bondarenko data for this nuclide object*/
  void setBondarenkoList(QList<BondarenkoData*> list)
  {
      /// provide hash function by mt
      for( int i = 0; i < list.size(); i++){
          addBondarenkoData(list[i]);
      }
  }

  /*! Obtain the BondarenkoData object with the given mt (reaction)
  * @param int mt - The reaction associated with the desired BondarenkoData
  * @return BondarenkoData  * - The BondarenkoData object desired, NULL if none associated with mt
   * */
  BondarenkoData * getBondarenkoDataByMt(int mt)
  {
      for( int i = 0; i < bondarenkoData.size(); i ++ ){
          if( bondarenkoData[i]->getMt() == mt ) return bondarenkoData[i];
      }
      return NULL;
  }

  /*! Obtain the Neutron1d data object at a given index
  * @return CrossSection1d * - The object requested, null if index is out of bounds
   * */
  CrossSection1d * getNeutron1dDataAt(int index)
  {
      if( index < 0 || index >= neutron1dData.size()) return NULL;
      return neutron1dData.value(index);
  }
  /*! Remove all Neutron1dData with a specified MT
  * @param int mt - the mt for which to remove by
  * @return int - The number of Neutron1dData removed
   * */
  int removeNeutron1dData(int mt)
  {
       int count = 0;
       for( int i =0; i < neutron1dData.size(); i ++){
           CrossSection1d * ptr = neutron1dData.value(i);
           if( ptr->getMt() == mt ){
                   neutron1dData.removeAt(i);
                   i--;
                   count++;
                   delete ptr;
           }
        }
       this->setNumNeutron1dData(neutron1dData.size());
       return count;
  }
  /*! Add a Neutron1dData object to this LibraryNuclide
  * If a Neutron1dData object with the same Mt already exists as
  * part of this LibraryNuclide, it will be removed (deleted) and replaced.
  * @param CrossSection1d * ptr - The pointer to the object to add
  * @return bool - true if Neutron1dData was succesfully added, false otherwise.
   * */
  bool addNeutron1dData(CrossSection1d * ptr)
  {
      if( ptr == NULL ) return false;
//      if( neutron1dDataByMt.contains(ptr->getMt())){
//          CrossSection1d * p2 = neutron1dDataByMt[ptr->getMt()];
//          printf("Nuclide %d has duplication reaction!\n",this->getId());
//          printf("Mt %d %d\n", ptr->getMt(), p2->getMt());
//          printf("sizes %d %d\n", ptr->getSize(), p2->getSize());
//          for( int i = 0; i < ptr->getSize(); i++)
//              printf("v[%d] %f %f\n", i, ptr->getAt(i), p2->getAt(i));
//      }
      neutron1dData.append(ptr);
      this->setNumNeutron1dData(neutron1dData.size());
      return true;
  }
  /*! Obtain the Neutron1dData object with the given mt (reaction)
  * @param int mt - The reaction associated with the desired Neutron1dData
  * @return CrossSection1d* - The Neutron1dData object desired, NULL if none associated with mt
   * */
  CrossSection1d * getNeutron1dDataByMt(int mt)
  {
      for( int i = 0; i < neutron1dData.size(); i ++){
          if( neutron1dData[i]->getMt() == mt ) return neutron1dData[i];
      }
      return NULL;
  }
  void setNeutron1dList(QList<CrossSection1d*> list)
  {
      /// provide hash function by mt
      for( int i = 0; i < list.size(); i++){
          addNeutron1dData(list[i]);
      }
  }
  QList<CrossSection1d *> getNeutron1dList(){return neutron1dData;}
  QList<CrossSection1d *> getGamma1dList(){return gamma1dData;}

  /*! Obtain the Gamma1d data object at a given index
  * @return CrossSection1d * - The object requested, null if index is out of bounds
   * */
  CrossSection1d * getGamma1dDataAt(int index)
  {
      if( index < 0 || index >= gamma1dData.size()) return NULL;
      return gamma1dData.value(index);
  }
  /*! Remove all Gamma1dData with a specified MT
  * @param int mt - the mt for which to remove by
  * @return int - The number of Gamma1dData removed
   * */
  int removeGamma1dData(int mt)
  {
       int count = 0;
       for( int i =0; i < gamma1dData.size(); i ++){
           CrossSection1d * ptr = gamma1dData.value(i);
           if( ptr->getMt() == mt ){
                   gamma1dData.removeAt(i);
                   i--;
                   count++;
                   delete ptr;
           }
       }
       this->setNumGamma1dData(gamma1dData.size());
       return count;
  }
  /*! Add a Gamma1dData object to this LibraryNuclide
  * If a Gamma1dData object with the same Mt already exists as
  * part of this LibraryNuclide, it will be removed (deleted) and replaced.
  * @param CrossSection1d * ptr - The pointer to the object to add
  * @return bool - true if Gamma1dData was succesfully added, false otherwise.*/
  bool addGamma1dData(CrossSection1d * ptr)
  {
      if( ptr == NULL ) return false;
      gamma1dData.append(ptr);
      this->setNumGamma1dData(gamma1dData.size());
      return true;
  }
  void setGamma1dList(QList<CrossSection1d*> list)
  {
      /// provide hash function by mt
      for( int i = 0; i < list.size(); i++){
          addGamma1dData(list[i]);
      }
  }
  /*! Obtain the Gamma1dData object with the given mt (reaction)
  /// @param int mt - The reaction associated with the desired Gamma1dData
  * @return CrossSection2d * - The Gamma1dData object desired, NULL if none associated with mt*/
  CrossSection1d * getGamma1dDataByMt(int mt)
  {
      for( int i = 0; i < gamma1dData.size(); i ++){
          if( gamma1dData[i]->getMt() == mt ) return gamma1dData[i];
      }
      return NULL;
  }
  QList<CrossSection2d*> getNeutron2dList(){return neutron2dData;}

  /*! Obtain the Neutron2d data object at a given index
  * @return CrossSection1d * - The object requested, null if index is out of bounds*/
  CrossSection2d * getNeutron2dDataAt(int index)
  {
      if( index < 0 || index >= neutron2dData.size()) return NULL;
      return neutron2dData.value(index);
  }
  /*! Remove all Neutron2dData with a specified MT
  * @param int mt - the mt for which to remove by
  * @return int - The number of Neutron2dData removed*/
  int removeNeutron2dData(int mt)
  {
      int count = 0;
      for( int i =0; i < neutron2dData.size(); i ++){
           CrossSection2d * ptr = neutron2dData.value(i);
           if( ptr->getMt() == mt ){
                   neutron2dData.removeAt(i);
                   i--;
                   count++;
                   delete ptr;
           }
       }
       this->setNumNeutron2dData(neutron2dData.size());
       return count;
  }
  /*! Add a Neutron2dData object to this LibraryNuclide
  * If a Neutron2dData object with the same Mt already exists as
  * part of this LibraryNuclide, it will be removed (deleted) and replaced.
  * @param CrossSection2d * ptr - The pointer to the object to add
  * @return bool - true if Neutron2dData was succesfully added, false otherwise.*/
  bool addNeutron2dData(CrossSection2d * ptr)
  {
      if( ptr == NULL ) return false;
      neutron2dData.append(ptr);
      this->setNumNeutron2dData(neutron2dData.size());

      return true;
  }
  void setNeutron2dList(QList<CrossSection2d*> list)
  {
      // provide hash function by mt
      for( int i = 0; i < list.size(); i++){
          addNeutron2dData(list[i]);
      }
  }
  /*! Obtain the Neutron2dData object with the given mt (reaction)
  * @param int mt - The reaction associated with the desired Neutron2dData
  * @return CrossSectio2d * - The Neutron2dData object desired, NULL if none associated with mt*/
  CrossSection2d * getNeutron2dDataByMt(int mt)
  {
      for( int i = 0; i < neutron2dData.size(); i ++){
          if( neutron2dData[i]->getMt() == mt ) return neutron2dData[i];
      }
      return NULL;
  }
  QList<CrossSection2d*> getGamma2dList(){return gamma2dData;}

  /*! Obtain the Gamma2d data object at a given index
  * @return CrossSection1d * - The object requested, null if index is out of bounds*/
  CrossSection2d * getGamma2dDataAt(int index)
  {
      if( index < 0 || index >= gamma2dData.size()) return NULL;
      return gamma2dData.value(index);
  }
  /*! Remove all Gamma2dData with a specified MT
  * @param int mt - the mt for which to remove by
  * @return int - The number of Gamma2dData removed*/
  int removeGamma2dData(int mt)
  {
       int count = 0;
       for( int i =0; i < gamma2dData.size(); i ++){
           CrossSection2d * ptr = gamma2dData.value(i);
           if( ptr->getMt() == mt ){
               gamma2dData.removeAt(i);
               i--;
               count++;
               delete ptr;
           }
       }
       this->setNumGamma2dData(gamma2dData.size());
       return count;
  }
  /*! Add a Gamma2dData object to this LibraryNuclide
  * If a Gamma2dData object with the same Mt already exists as
  * part of this LibraryNuclide, it will be removed (deleted) and replaced.
  * @param CrossSection2d * ptr - The pointer to the object to add
  * @return bool - true if Gamma2dData was succesfully added, false otherwise.*/
  bool addGamma2dData(CrossSection2d * ptr)
  {
      if( ptr == NULL ) return false;
      gamma2dData.append(ptr);
      this->setNumGamma2dData(gamma2dData.size());
      return true;
  }
  void setGamma2dList(QList<CrossSection2d*> list)
  {
      // provide hash function by mt
      for( int i = 0; i < list.size(); i++){
          addGamma2dData(list[i]);
      }
  }
  /*! Obtain the Gamma2dData object with the given mt (reaction)
  * @param int mt - The reaction associated with the desired Gamma2dData
  * @return CrossSectio2d * - The Gamma2dData object desired, NULL if none associated with mt*/
  CrossSection2d * getGamma2dDataByMt(int mt)
  {
      for( int i = 0; i < gamma2dData.size(); i ++){
          if( gamma2dData[i]->getMt() == mt ) return gamma2dData[i];
      }
      return NULL;
  }
  QList<CrossSection2d*> getGammaProdList(){return gammaProdData;}

  /*! Obtain the GammaProd data object at a given index
  * @return CrossSection1d * - The object requested, null if index is out of bounds*/
  CrossSection2d * getGammaProdDataAt(int index)
  {
      if( index < 0 || index >= gammaProdData.size()) return NULL;
      return gammaProdData.value(index);
  }
  /*! Remove all GammaProdData with a specified MT
  * @param int mt - the mt for which to remove by
  * @return int - The number of GammaProdData removed*/
  int removeGammaProdData(int mt)
  {
       int count = 0;
       for( int i =0; i < gammaProdData.size(); i ++){
           CrossSection2d * ptr = gammaProdData.value(i);
           if( ptr->getMt() == mt ){
               gammaProdData.removeAt(i);
               i--;
               count++;
               delete ptr;
           }
       }
       this->setNumGammaProdData(gammaProdData.size());
       return count;
  }
  /*! Add a GammaProdData object to this LibraryNuclide
  * If a GammaProdData object with the same Mt already exists as
  * part of this LibraryNuclide, it will be removed (deleted) and replaced.
  * @param CrossSection2d * ptr - The pointer to the object to add
  * @return bool - true if GammaProdData was succesfully added, false otherwise.*/
  bool addGammaProdData(CrossSection2d * ptr)
  {
      if( ptr == NULL ) return false;
      gammaProdData.append(ptr);
      this->setNumGammaProdData(gammaProdData.size());
      return true;
  }
  void setGammaProdList(QList<CrossSection2d*> list)
  {
      // provide hash function by mt
      for( int i = 0; i < list.size(); i++){
          addGammaProdData(list[i]);
      }

  }
  /*! Obtain the GammaProdData object with the given mt (reaction)
  * @param int mt - The reaction associated with the desired GammaProdData
  * @return CrossSectio2d * - The GammaProdData object desired, NULL if none associated with mt*/
  CrossSection2d * getGammaProdDataByMt(int mt)
  {
      for( int i = 0; i < gammaProdData.size(); i ++){
          if( gammaProdData[i]->getMt() == mt ) return gammaProdData[i];
      }
      return NULL;
  }
  NuclideResonance * getResonance(){return resonance;}
  void setResonance(NuclideResonance * resonance){this->resonance = resonance;}

  /*! Obtain the description of this nuclide
  * @return char * - character array containing the description */
  char * getDescription(){ return d->description;}

  /*! set the nuclide's description*/
  void setDescription(const char * desc){
      if( desc == NULL ) return;
      qAtomicDetach(d);
      strncpy(this->d->description,desc, AMPX_NUCLIDE_DESCR_LENGTH);
      this->d->description[AMPX_NUCLIDE_DESCR_LENGTH] = '\0';
  }// end of setDescription

  /*! Obtain the data member in the nuclide
  * these data members are id, numNeu1dData, etc
  * and have enumerated definition
  * AMPX_MAST_NUCLIDE_ID, _NUM_BOND_SETS, etc.
  * you can also refer by word enumeration, which have the following
  * AMPX_NUCLIDE_WORD_19, _37, etc.*/
  int getData(int var){
      if( var < 0 || var >= AMPX_NUMBER_NUCLIDE_INTEGER_OPTIONS ) return 0;
      return d->words[var];
  }
  void setData(int var, int value){
      if( var < 0 || var >= AMPX_NUMBER_NUCLIDE_INTEGER_OPTIONS ) return ;
      qAtomicDetach(d);
      d->words[var]=value;
  }
  float getDataAsFloat(int var){
      if( var < 0 || var >= AMPX_NUMBER_NUCLIDE_INTEGER_OPTIONS ) return 0;
      int original = d->words[var];
      float *value = reinterpret_cast<float*>(&original);
      return *value;
  }
  void setDataAsFloat(int var, float value){
      int *iValue = reinterpret_cast<int *>(&value);
      setData(var,*iValue);
  }
  /*! enumerate all data options*/
  virtual int getId(){return getData(AMPX_MAST_NUCLIDE_ID);}
  virtual void setId(int id);
  virtual int getNumResolvedResonance(){return getData(AMPX_MAST_NUCLIDE_NUM_RESOLVED_RESDATA);}
  virtual void setNumResolvedResonance(int num){setData(AMPX_MAST_NUCLIDE_NUM_RESOLVED_RESDATA,num);}
  virtual int getNumEnergiesUnresolvedResonance(){return getData(AMPX_MAST_NUCLIDE_NUM_ENERGIES_UNRESOLVED_RESDATA);}
  virtual void setNumEnergiesUnresolvedResonance(int num){setData(AMPX_MAST_NUCLIDE_NUM_ENERGIES_UNRESOLVED_RESDATA,num);}
  virtual int getNumNeutron1dData(){return getData(AMPX_MAST_NUCLIDE_NUM_NEU_1D_DATA);}
  virtual void setNumNeutron1dData(int num){setData(AMPX_MAST_NUCLIDE_NUM_NEU_1D_DATA,num);}
  virtual int getNumNeutron2dData(){return getData(AMPX_MAST_NUCLIDE_NUM_NEU_2D_DATA);}
  virtual void setNumNeutron2dData(int num){setData(AMPX_MAST_NUCLIDE_NUM_NEU_2D_DATA,num);}
  virtual int getNumGamma1dData(){return getData(AMPX_MAST_NUCLIDE_NUM_GAM_1D_DATA);}
  virtual void setNumGamma1dData(int num){setData(AMPX_MAST_NUCLIDE_NUM_GAM_1D_DATA,num);}
  virtual int getNumGamma2dData(){return getData(AMPX_MAST_NUCLIDE_NUM_GAM_2D_DATA);}
  virtual void setNumGamma2dData(int num){setData(AMPX_MAST_NUCLIDE_NUM_GAM_2D_DATA,num);}
  virtual int getNumGammaProdData(){return getData(AMPX_MAST_NUCLIDE_NUM_NEU2GAM_2D_DATA);}
  virtual void setNumGammaProdData(int num){setData(AMPX_MAST_NUCLIDE_NUM_NEU2GAM_2D_DATA,num);}
  virtual float getAtomicMass(){return getDataAsFloat(AMPX_MAST_NUCLIDE_A);}
  virtual void setAtomicMass(float am){setDataAsFloat(AMPX_MAST_NUCLIDE_A,am);}
  virtual int getZA(){return getData(AMPX_MAST_NUCLIDE_ZA);}
  virtual void setZA(int za){setData(AMPX_MAST_NUCLIDE_ZA,za);}
  virtual int getMixture(){return getData(AMPX_NUCLIDE_MIXTURE);}
  virtual void setMixture(int mix);
  virtual float getPowerPerFission(){return getDataAsFloat(AMPX_MAST_NUCLIDE_POWER_PER_FISSION);}
  virtual void setPowerPerFission(float ppf){setDataAsFloat(AMPX_MAST_NUCLIDE_POWER_PER_FISSION,ppf);}
  virtual float getEnergyReleasePerCapture(){return getDataAsFloat(AMPX_MAST_NUCLIDE_ENERGY_RELEASE_PER_CAP);}
  virtual void setEnergyReleasePerCapture(float rpc){setDataAsFloat(AMPX_MAST_NUCLIDE_ENERGY_RELEASE_PER_CAP,rpc);}
  virtual int getMaxLengthScatterArray(){return getData(AMPX_MAST_NUCLIDE_MAX_LENGTH_SCATTER_MATRIX);}
  virtual void setMaxLengthScatterArray(int num){setData(AMPX_MAST_NUCLIDE_MAX_LENGTH_SCATTER_MATRIX,num);}

  virtual int getNumBondarenkoData(){return getData(AMPX_MAST_NUCLIDE_NUM_BOND_SETS);}
  virtual void setNumBondarenkoData(int num){setData(AMPX_MAST_NUCLIDE_NUM_BOND_SETS,num);}
  virtual int getMaxNumBondGrp(){return getData(AMPX_MAST_NUCLIDE_MAX_NUM_BOND_GRPS);}
  virtual void setMaxNumBondGrp(int num){setData(AMPX_MAST_NUCLIDE_MAX_NUM_BOND_GRPS,num);}
  virtual int getNumBondarenkoSig0Data(){return getData(AMPX_MAST_NUCLIDE_NUM_BOND_SIG_SETS);}
  virtual void setNumBondarenkoSig0Data(int num){setData(AMPX_MAST_NUCLIDE_NUM_BOND_SIG_SETS,num);}
  virtual int getNumBondarenkoTempData(){return getData(AMPX_MAST_NUCLIDE_NUM_BOND_TEMP_SETS);}
  virtual void setNumBondarenkoTempData(int num){setData(AMPX_MAST_NUCLIDE_NUM_BOND_TEMP_SETS,num);}
  virtual float getPotentialScatter(){return getDataAsFloat(AMPX_MAST_NUCLIDE_POTEN_SCATTER);}
  virtual void setPotentialScatter(float ps){setDataAsFloat(AMPX_MAST_NUCLIDE_POTEN_SCATTER,ps);}
  virtual int getFastNeutronData(){return getData(AMPX_MAST_NUCLIDE_FAST_NEU_DATA);}
  virtual void setFastNeutronData(int ps){setData(AMPX_MAST_NUCLIDE_FAST_NEU_DATA,ps);}
  virtual int getThermalNeutronData(){return getData(AMPX_MAST_NUCLIDE_THERM_NEU_DATA);}
  virtual void setThermalNeutronData(int ps){setData(AMPX_MAST_NUCLIDE_THERM_NEU_DATA,ps);}
  virtual int getGammaData(){return getData(AMPX_MAST_NUCLIDE_GAM_DATA);}
  virtual void setGammaData(int ps){setData(AMPX_MAST_NUCLIDE_GAM_DATA,ps);}
  virtual int getGammaProdData(){return getData(AMPX_MAST_NUCLIDE_GAMPROD_DATA);}
  virtual void setGammaProdData(int ps){setData(AMPX_MAST_NUCLIDE_GAMPROD_DATA,ps);}
  virtual int getSourceId(){return getData(AMPX_NUCLIDE_SOURCE);}
  virtual void setSourceId(int s){setData(AMPX_NUCLIDE_SOURCE,s);}
  virtual int getNumRecords(){return getData(AMPX_MAST_NUCLIDE_NUM_RECORD_SETS);}
  virtual void setNumRecords(int numRecords){return setData(AMPX_MAST_NUCLIDE_NUM_RECORD_SETS, numRecords);}
  virtual int getSampleNo(){return getData(AMPX_MAST_SAMPLE_NO);}
  virtual void setSampleNo(int id){
      setData(AMPX_MAST_SAMPLE_NO,id);
  }
  
  /// obtain a string representation of this nuclide
  QString toQString(){
      QString value = QString("Description=(%1)\n")
                       .arg(QString(getDescription()).trimmed());
      for( int i = 0; i < AMPX_NUMBER_NUCLIDE_INTEGER_OPTIONS; i++ ){
          float fVal = getDataAsFloat(i);
          int iVal = getData(i);
          bool isFloat = false;
          switch(i){
              case AMPX_MAST_NUCLIDE_A:
              ///case AMPX_MAST_NUCLIDE_ZA:
              case AMPX_MAST_NUCLIDE_POWER_PER_FISSION:
              case AMPX_MAST_NUCLIDE_ENERGY_RELEASE_PER_CAP:
              case AMPX_MAST_NUCLIDE_POTEN_SCATTER:
              ///case AMPX_MAST_NUCLIDE_FAST_NEU_DATA:
              ///case AMPX_MAST_NUCLIDE_THERM_NEU_DATA:
              ///case AMPX_MAST_NUCLIDE_GAMPROD_DATA:
              ///case AMPX_MAST_NUCLIDE_GAM_DATA:
                   isFloat = true;
                   break;
              default:
                   break;
          }
          value += QString("  word[%1]=%2 isFloat=%3\n")
                          .arg(i+AMPX_NUCLIDE_INTEGER_OPTION_START)
                          .arg(isFloat ? fVal : iVal).arg(isFloat);
      }
      return value;
  }

    // serialization interface

    /**
     * @brief Serialize the object into a contiguous block of data
     * @param Standard::AbstractStream * stream - the stream into which the contiguous data will be stored
     * @return int - 0 upon success, error otherwise
     */
    //virtual int serialize(Standard::AbstractStream * stream) const;

    /**
     * @brief deserialize the object from the given Standard::AbstractStream
     * @param Standard::AbstractStream * stream - the stream from which the object will be inflated
     * @return int - 0 upon success, error otherwise
     */
    //virtual int deserialize(Standard::AbstractStream * stream);
    /**
     * @brief Obtain the size in bytes of this object when serialized
     * @return unsigned long - the size in bytes of the object when serialized
     */
    //virtual unsigned long getSerializedSize() const;
    /**
     * @brief Obtain the universal version identifier
     * the uid should be unique for all Standard::Serializable objects such that
     * an object factory can retrieve prototypes of the desired Standard::Serializable
     * object and inflate the serial object into a manageable object
     * @return long - the uid of the object
     */
    virtual long getUID() const;
    /**
     * @brief Determine if this object is a child of the given parent UID
     * This is intended to assist in object deserialization, when an object
     * may have been subclassed, the appropriate subclass prototype must be
     * obtained from the object factory in order to correctly deserialize and
     * validate object inheritance.
     * i.e.
     * Object X contains a list of object Y. Object Y has child class object Z.
     * When deserializing object X, we expect Y, or a child of Y to be
     * deserialized, any other objects would be an error.
     * @return bool - true, if this object is a child of a class with the given uid, false otherwise
     */
    //virtual bool childOf(long parentUID) const {return Standard::Serializable::getUID()==parentUID || Standard::Serializable::childOf(parentUID);}
    
    /**
     * Get a string that describes the source of the data, i.e. endf, jeff, etc
     * @param value the string in which to save the source information
     */
    virtual void getSourceAsString( std::string & value){
        int i = getSourceId();       
        ScaleData::LibrarySourceDefs::getLibrarySourceAsString(i, value);          
    }
    
    std::string getSourceLibrary(){
        std::string tmp;
        int i = getSourceId();       
        ScaleData::LibrarySourceDefs::getLibrarySourceAsString(i, tmp);
        return tmp;
    }
public:
    /**
     * @brief the universal version identifier for this class
     */
    static const long uid;
};
#endif
