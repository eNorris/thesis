#include "LibraryItem.h"
#include "resources.h"
#include <QString>

class NuclideResonance : public LibraryItem
{
private:
    float * resolved, * unresolved;
    int resolvedSize, unresolvedSize;
    void initialize();
public:
    NuclideResonance();
    ~NuclideResonance();
    NuclideResonance(const NuclideResonance & orig);
    
    float * getResolved(){return resolved;}
    int getResolvedSize(){return resolvedSize;}
    void setResolved(float * resolved, int size){this->resolved=resolved;resolvedSize=size;}
    float * getUnresolved(){return unresolved;}
    int getUnresolvedSize(){return unresolvedSize;}
    void setUnresolved(float * unresolved, int size){this->unresolved=unresolved;unresolvedSize=size;}
    
    QString toQString(){
        QString results = QString("Nuclide Resonance\n");
        if( getResolved() != NULL ){
            int length = getResolvedSize();
            for( int i = 0; i < length; i++ ){
                results += QString("  resolved[%1]=%2\n").arg(i).arg(getResolved()[i]);
            }
        }
        if( getUnresolved() != NULL ){
            int length = getUnresolvedSize();
            for( int i = 0; i < length; i++ ){
                results += QString("  unresolved[%1]=%2\n").arg(i).arg(getUnresolved()[i]);
            }
        }
        return results;
    }// end of toQString   
    
}; 
