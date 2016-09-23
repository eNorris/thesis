//#include "Resource/AmpxLib/ampxlib_config.h"
#include "LibraryHeader.h"
#include <stdio.h>
#include <string.h>
#include <QDebug>
#include <QString>
//#include "Standard/Interface/AbstractStream.h"
LibraryHeader::LibraryHeader()
{
    initialize();
}

void LibraryHeader::initialize()
{
     idtape = 0;
     nnuc = 0;
     igm = 0;
     iftg = 0;
     msn = 0;
     ipm = 0;
     i1 = 0;
     i2 = 0;
     i3 = 0;
     i4 = 0;
     strcpy(itm,"");
}

bool LibraryHeader::operator==(LibraryHeader & a){
    if( QString(itm).trimmed() !=  QString(a.itm).trimmed() ) return false;
    if( idtape != a.idtape )return false;
    if( nnuc != a.nnuc ) return false;
    if( igm != a.igm ) return false;
    if( iftg != a.iftg ) return false;
    if( msn != a.msn ) return false;
    if( ipm != a.ipm ) return false;
    if( i1 != a.i1 ) return false;
    if( i2 != a.i2 ) return false;
    if( i3 != a.i3 ) return false;
    if( i4 != a.i4 ) return false;
    return true;
}
// serializable interface
const long LibraryHeader::uid = 0x39513254f6b0dc7e;

