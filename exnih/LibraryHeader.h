#ifndef LIBRARY_HEADER_H
#define LIBRARY_HEADER_H
#include <QString>
#include <string.h>
//#include "Standard/Interface/Serializable.h"
#include "LibraryItem.h"

using namespace std;

#define AMPX_HEADER_TITLE_LENGTH (4*100)
#define AMPX_HEADER_NUMBER_INTEGER_OPTIONS 10

class LibraryHeader : public LibraryItem{

private:
  int idtape;    /// identification number of tape
  int nnuc;     /// number of data sets on the tape
  int igm;       /// number of neutron groups
  int iftg;      /// the first thermal neutron group on the library
  int msn;       /// 0 - pre-NITWAL-III library 1 - library for use with NITWAL-II        I
  int ipm;       /// number of gamma groups
  int i1;        /// zero
  int i2;        /// working library flag
  int i3;        /// zero
  int i4;        /// zero
  char itm[AMPX_HEADER_TITLE_LENGTH+1]; /// title

  void initialize();
public:
  LibraryHeader();

  bool operator==(LibraryHeader & a);

  LibraryHeader * getCopy() const{
      LibraryHeader * copy = new LibraryHeader();
      copy->idtape = idtape;
      copy->nnuc = nnuc;
      copy->igm = igm;
      copy->iftg = iftg;
      copy->msn = msn;
      copy->ipm = ipm;
      copy->i1 = i1;
      copy->i2 = i2;
      copy->i3 = i3;
      copy->i4 = i4;
      strncpy(copy->itm, itm, AMPX_HEADER_TITLE_LENGTH);
      return copy;
  }

  int getIdTape(){return idtape;}
  void setIdTape(int idtape){this->idtape = idtape;}
  int getNNuc(){return nnuc;}
  void setNNuc(int nnuc){this->nnuc = nnuc;}
  int getIGM(){return igm;}
  void setIGM(int igm){this->igm = igm;}
  int getIFTG(){return iftg;}
  void setIFTG(int iftg){this->iftg = iftg;}
  int getMSN(){return msn;}
  void setMSN(int msn){this->msn = msn;}
  int getIPM(){return ipm;}
  void setIPM(int ipm){this->ipm = ipm;}
  int getI1(){return i1;}
  void setI1(int i1){this->i1 = i1;}
  int getI2(){return i2;}
  void setI2(int i2){this->i2 = i2;}
  int getI3(){return i3;}
  void setI3(int i3){this->i3 = i3;}
  int getI4(){return i4;}
  void setI4(int i4){this->i4 = i4;}
  char * getITM(){ return itm;}
  void setITM(char * itm){
      if( itm == NULL ) return;
      strncpy(this->itm,itm,AMPX_HEADER_TITLE_LENGTH);
      this->itm[AMPX_HEADER_TITLE_LENGTH] = '\0';
  }
  QString toQString(){
      return (QString("id=%1 nnuc=%2 igm=%3 iftm=%4 msn=%5")+
                       QString(" ipm=%6 i1=%7 i2=%8 i3=%9 i4=%10\nitm=%11\n"))
                       .arg(getIdTape())
                       .arg(getNNuc())
                       .arg(getIGM())
                       .arg(getIFTG())
                       .arg(getMSN())
                       .arg(getIPM())
                       .arg(getI1())
                       .arg(getI2())
                       .arg(getI3())
                       .arg(getI4())
                       .arg(getITM());
  }

public:
    /**
     * @brief the universal version identifier for this class
     */
    static const long uid;
};

#endif
