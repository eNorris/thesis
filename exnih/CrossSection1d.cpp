#include "CrossSection1d.h"
//#include "Resource/AmpxLib/ampxlib_config.h"
//#include "Standard/Interface/AbstractStream.h"
CrossSection1d::CrossSection1d()
{
    initialize();
}
CrossSection1d::CrossSection1d(const CrossSection1d & orig){
    d = orig.d;
    d->ref.ref();
}
void CrossSection1d::initialize()
{
    d = new CrossSection1dData();
}

bool CrossSection1d::operator==(CrossSection1d & a){
    if( d == a.d ) return true;
    if( getMt() != a.getMt()) return false;
    if( getSize() != a.getSize() ) return false;
    for( int i = 0; i < getSize(); i++){
        if( getAt(i) != a.getAt(i) ) return false;
    }
    return true;
}
CrossSection1d::~CrossSection1d()
{
    if( !d->ref.deref() ){
        delete d;
    }
}

CrossSection1d * CrossSection1d::getCopy() const
{
    CrossSection1d * copy = new CrossSection1d(*this);
    return copy;
}

void CrossSection1d::setMt(int mt)
{
    int oldMt = getMt();
    if( oldMt == mt ) return;
    qAtomicDetach(d);
    this->d->mt=mt;
}
// Serialization interfaces

const long CrossSection1d::uid = 0xabeef95102040602;

