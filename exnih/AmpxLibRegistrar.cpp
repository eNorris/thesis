#include <stdio.h>
#include "Standard/Interface/SerialFactory.h"
#include "AmpxLibRegistrar.h"
#include "BondarenkoData.h"
#include "BondarenkoInfiniteDiluted.h"
#include "BondarenkoFactors.h"
#include "BondarenkoGlobal.h"
#include "SinkGroup.h"
#include "ScatterMatrix.h"
#include "CrossSection1d.h"
#include "CrossSection2d.h"
#include "LibraryNuclide.h"
#include "WorkingLibraryNuclide.h"
#include "LibraryHeader.h"
#include "LibraryEnergyBounds.h"
#include "AmpxLibrary.h"
#include "SubGroupData.h"
/**
 * @brief AmpxLibrary registrar for all serializable AmpxLib objects
 * @param SerialFactory * factory - the factory to which AmpxLib objects are registered
 */
void AmpxLibRegistrar( Standard::SerialFactory * factory){
    if( factory == NULL) return;
    factory->registerSerializable(new BondarenkoData());
    factory->registerSerializable(new BondarenkoInfiniteDiluted());
    factory->registerSerializable(new BondarenkoFactors());
    factory->registerSerializable(new BondarenkoGlobal());
    factory->registerSerializable(new SinkGroup());
    factory->registerSerializable(new ScatterMatrix());
    factory->registerSerializable(new CrossSection1d());
    factory->registerSerializable(new CrossSection2d());
    factory->registerSerializable(new LibraryNuclide());
    factory->registerSerializable(new WorkingLibraryNuclide());
    factory->registerSerializable(new LibraryHeader());
    factory->registerSerializable(new LibraryEnergyBounds());
    factory->registerSerializable(new AmpxLibrary());
    
    // subgroup data
    factory->registerSerializable(new SubGroupWeight());
    factory->registerSerializable(new SubGroupSetGrp());
    factory->registerSerializable(new SubGroupSet());
    factory->registerSerializable(new SubGroupData());
}
