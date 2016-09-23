#ifndef SUBGROUPDATA_H
#define	SUBGROUPDATA_H

#include "CrossSection1d.h"
#include <QObject>
#include <QList>
//#include "Standard/Interface/AbstractStream.h"

class SubGroupWeight:public CrossSection1d{
  public:
    SubGroupWeight():CrossSection1d(),temp(0.0){}
    SubGroupWeight(const SubGroupWeight& orig):CrossSection1d(orig){
        temp = orig.temp;
    }
    virtual ~SubGroupWeight(){}

     virtual SubGroupWeight * getCopy() const {
         return new SubGroupWeight(*this);
     }

    /**
     * Get the temperature for this weight set
     * @return  the temperature of for this weight set
     */
    float getTemp()const{
        return temp;
    }

    /**
     * Set the temperature for this weight set
     * @return  the temperature of for this weight set
     */
    void setTemp(double temp){
        this->temp = temp;
    }

    bool operator==(SubGroupWeight & a);

    static const long uid;



  private:
      /** The temperature for this weight set */
      float temp;
};




class SubGroupSetGrp { //Standard::Serializable{
   public:
      SubGroupSetGrp(){}//:Serializable(){}
      SubGroupSetGrp(const SubGroupSetGrp& orig);
      virtual ~SubGroupSetGrp();

      SubGroupSetGrp * getCopy() const {
         return new SubGroupSetGrp(*this);
     }

      /**
       * Get the group number for this subgroup
       * @return  the group number for this subgroup
       */
      int getGrp() const{
          return ig;
      }

      /**
       * Set the group number for this subgroup
       * @param  grp the group number for this subgroup
       */
      void setGrp(int grp) {
          ig = grp;
      }


      /**
       * Get the list of available temperatures.
       * This includes the weight data only
       * @return  the list of available temperatures.
       */
      void  addTemperatures(QList<float> * temps) const;


      /**
       * Get the list of available reactions.
       * This includes cross section data as well as weight data
       * @return  the list of available reaction.
       */
      void addMts(QList<int> * mts) const;



      /**
       * Get the cross section data  for the indicated mt
       * @param mt the desired mt value
       * @return  the cross section data for the desired mt or 0 if not found
       */
      CrossSection1d * getCrossByMt(int mt);



      /**
       * Get the weight data  for the indicated mt and temperature
       * @param mt the desired mt value
       * @param temp the desired temperature
       * @return  the weight data  for the indicated mt and temperature  or 0 if not found
       */
      SubGroupWeight * getWeightByMt(int mt, float temp);

      /**
       * Check whether the given cross section data exist
       * @param mt the desired mt value
       * @return
       */
      bool containsCrossByMt(int mt);


      /**
       * Return the cross section at the indicated index
       * @param index the desired index
       * @return
       */
      CrossSection1d * getCrossAt(int index){
          if( index < 0 || index >= cross.size()) return 0;
          return cross[index];
      }

      /**
       * Return the  weight at the indicated index
       * @param index the desired index
       * @return
       */
      SubGroupWeight * getWeightAt(int index){
          if( index < 0 || index >= weight.size()) return 0;
          return weight[index];
      }


      /**
       * Get the number of cross section data
       * @return  the number of cross section data
       */
      int getNumCross() const{
          return cross.size();
      }


      /**
       * Get the number of weights
       * @return
       */
      int getNumWeight() const{
         return weight.size();
      }




      /**
       * Add new cross section data
       * @param data the cross section data to add
       */
      void addCrossSection(CrossSection1d * data){
          cross.append(data);
      }


      /**
       * Add new weight data
       * @param data the weight data to add
       */
      void addWeight(SubGroupWeight * data){
          weight.append(data);
      }

      bool operator==(SubGroupSetGrp & a);

       static const long uid;


  private:
      /** the subgroup number */
      int ig;

      /** The list of cross section data */
      QList<CrossSection1d *>  cross;

      /** The list of weights*/
      QList<SubGroupWeight *>  weight;
};


class SubGroupSet{
    public:
      SubGroupSet(){
          groups = 0;
          ngrp = 0;
      }
      SubGroupSet(const SubGroupSet& orig);
      virtual ~SubGroupSet();

      SubGroupSet * getCopy() const{
         return new SubGroupSet(*this);
      }

      /**
       * Sets the size for this group set
       * @param size
       */
      void setSize(int size);

      int getSize(){
          return ngrp;
      }

      /**
       * Get the subgroupset by index
       * @param index the desired index
       * @return  the SubGroupSetGrp at the desired index or 0 if it does not exist
       */
     SubGroupSetGrp * getSubByIndex(int index){
         if( index < 0 || index >= ngrp)  return 0;
         return groups[index];
     }


     /**
       * Get the subgroupset by group
       * @param index the desired group
       * @return  the SubGroupSetGrp at the desired group or 0 if it does not exist
       */
     SubGroupSetGrp * getSubByGrp(int grp){
         for( int i = 0; i < ngrp; i++){
             if( groups[i] != 0 && groups[i]->getGrp() == grp) return groups[i];
         }
         return 0;
     }

     /**
      * Add a new subgroup
      * @param grp  the group to add
      * @param index the index
      */
     void addSubGroup(SubGroupSetGrp * grp, int index){
         if( index < 0 || index >= ngrp) return;
         if( groups[index] != 0) delete groups[index];
         groups[index] =  grp;
     }

     /**
       * Get the list of available reactions.
       * This includes cross section data as well as weight data
       * @return  the list of available reaction.
       */
     void addMts(QList<int> * mts) const;


       /**
       * Get the list of available temperatures.
       *
       * @return  the list of available temperatures.
       */
      void  addTemperatures(QList<float> * temps) const;

      /**
       * Get the cross section data  for the indicated mt and sub group
       * @param mt the desired mt value
       * @param ig the desired subgroup
       * @return  the cross section data for the desired mt  and subgroup or 0 if not found
       */
      CrossSection1d * getCrossByMt(int mt, int ig);


       /**
       * Get the weight data  for the indicated mt and temperature
       * @param mt the desired mt value
       * @param temp the desired temperature
       *  @param ig the desired subgroup
       * @return  the weight data  for the indicated mt and temperature  or 0 if not found
       */
      SubGroupWeight * getWeightByMt(int mt, float temp, int ig);

      bool operator==(SubGroupSet & a);



      static const long uid;


  private:
    /** The subgroup group data */
    SubGroupSetGrp ** groups;

    /** The number of groups */
    int ngrp;
};


class SubGroupData{
public:

    SubGroupData(){}
    SubGroupData(const SubGroupData& orig);
    virtual ~SubGroupData();

     SubGroupData * getCopy() const{
         return new SubGroupData(*this);
     }

    /**
     * Get the first group for which sub-group data are available
     * @return  first group for which sub-group data are available
     */
    int getNeutStart() const{
        return neutStart;
    }

    /**
     * Get the last group for which sub-group data are available
     * @return  last group for which sub-group data are available
     */
    int getNeutEnd() const{
        return neutEnd;
    }



     // setters
     /**
     * Set the first group for which sub-group data are available
     * @param nstart  first group for which sub-group data are available
     */
    void setNeutStart(int nstart) {
        neutStart = nstart;
    }

    /**
     * Set the last group for which sub-group data are available
     * @param nstart  last group for which sub-group data are available
     */
   void setNeutEnd(int nend) {
        neutEnd = nend;
    }



     /**
      * Add a new sub group
      * @param set
      */
     void addSubGroupSet(SubGroupSet * set){
         subGrps.append(set);
     }

     /**
      * Get the number of subgroup sets
      * @return the number of subgroup sets
      */
     int getNumSubSets() const{
         return subGrps.size();
     }

     /**
      * Get the subgroup set at the indicated index
      * @param index the index for the sub group set
      * @return
      */
     SubGroupSet * getSubGrpSet(int index){
         if( index < 0 && index >= subGrps.size()) return 0;
         return subGrps[index];
     }

     bool operator==(SubGroupData & a);

       static const long uid;


private:
    /** The first group for which sub-group data are available    */
    int neutStart;

    /**  The  last group for which sub-group data are available   */
    int neutEnd;


    /** The list of cross section data */
    QList< SubGroupSet*>  subGrps;

};

#endif	/* SUBGROUPDATA_H */
