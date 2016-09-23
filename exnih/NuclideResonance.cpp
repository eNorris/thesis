#include "NuclideResonance.h"
NuclideResonance::NuclideResonance()
{
    initialize();
}
void NuclideResonance::initialize()
{
    setResolved(NULL,0);
    setUnresolved(NULL,0);
}
NuclideResonance::~NuclideResonance()
{
    if( getResolved() != NULL ){
        delete[] getResolved();
        setResolved(NULL,0);
    }
    if( getUnresolved() != NULL ){
        delete[] getUnresolved();
        setUnresolved(NULL,0);
    }
}

NuclideResonance::NuclideResonance(const NuclideResonance & orig){
    resolvedSize = orig.resolvedSize;
    unresolvedSize  = orig.unresolvedSize;
    resolved = new float[resolvedSize];
    for( int i = 0; i < resolvedSize; i++) resolved[i] = orig.resolved[i];
    unresolved = new float[unresolvedSize];
    for( int i = 0; i < unresolvedSize; i++) unresolved[i] = orig.unresolved[i];
}