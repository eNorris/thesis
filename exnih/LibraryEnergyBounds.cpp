//#include "Resource/AmpxLib/ampxlib_config.h"
//#include "Standard/Interface/AbstractStream.h"
#include "LibraryEnergyBounds.h"
#include "resources.h"
LibraryEnergyBounds::LibraryEnergyBounds()
{
    initialize();
}
void LibraryEnergyBounds::initialize()
{
}

LibraryEnergyBounds::~LibraryEnergyBounds()
{
}
LibraryEnergyBounds * LibraryEnergyBounds::getCopy() const
{
    LibraryEnergyBounds * copy = new LibraryEnergyBounds();
    if( this->getBoundsSize() > 0 ) copy->setBounds(getBounds(), getBoundsSize());
    if( this->getLethargySize() > 0 ) copy->setLethargyBounds(getLethargyBounds(), getLethargySize());

    return copy;
}
bool LibraryEnergyBounds::operator==(LibraryEnergyBounds & a){
    if( bounds.size() != a.bounds.size() ) return false;

    for( size_t i = 0; i < bounds.size(); i++){
        if( bounds[i] != a.bounds[i] ) return false;
    }

    if( lethargyBounds.size() != a.lethargyBounds.size() ) return false;

    for( size_t i = 0; i < lethargyBounds.size(); i++){
        if( lethargyBounds[i] != a.lethargyBounds[i] ) return false;
    }
    return true;
}
// serializable interface
const long LibraryEnergyBounds::uid = 0x931523456f0bdc6a;


