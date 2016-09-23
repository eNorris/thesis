/*
 * File:   AmpxDataHelper.h
 * Author: raq
 *
 * Created on April 2, 2012, 1:36 PM
 */

#ifndef AMPXDATAHELPER_H
#define	AMPXDATAHELPER_H
#include <QList>
class LibraryNuclide;
class CrossSection1d;
class CrossSection2d;
class BondarenkoData;
class BondarenkoGlobal;
class AmpxLibrary;

/**
 * @class AmpxDataHelper
 * @brief This class is to assist with retrieval of data from an ampx data component( nuclide, etc)
 * Most methods provided have take a data component (nuclide, etc) and an AmpxLibrary.
 * If the data requested is not on the nuclide, the methods will consult the ampx library.
 */
class AmpxDataHelper {
public:
    AmpxDataHelper();
    AmpxDataHelper(const AmpxDataHelper& orig);
    virtual ~AmpxDataHelper();
private:

public:
/**
    * @brief Obtain the nuclide description
    * @param nuclide - the nuclide for which to obtain the NumRecords
    * @param library - the library that backs up the nuclide
    * @return char * - the character array containing the description
    * Note: This checks the nuclide to determine if the nuclide has the description
    * if the nuclide's description is NULL, it attempts to retrieve the value from the nuclide
    * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably NULL) from the
    * given nuclide will be returned.
    */
    static char * getDescription(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the NumRecords
     * @param nuclide - the nuclide for which to obtain the NumRecords
     * @param library - the library that backs up the nuclide
     * @return int - the NumRecords
     * Note: This check the nuclide to determine if the nuclide has the NumRecords
     * if the nuclide's NumRecords is zero, it attempts to retrieve the value from the nuclide
     * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
     * given nuclide will be returned.
     */
    static int getNumRecords(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the GammaProdData
     * @param nuclide - the nuclide for which to obtain the GammaProdData
     * @param library - the library that backs up the nuclide
     * @return int - the GammaProdData
     * Note: This check the nuclide to determine if the nuclide has the GammaProdData
     * if the nuclide's GammaProdData is zero, it attempts to retrieve the value from the nuclide
     * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
     * given nuclide will be returned.
     */
    static int getGammaProdData(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the GammaData
     * @param nuclide - the nuclide for which to obtain the GammaData
     * @param library - the library that backs up the nuclide
     * @return int - the GammaData
     * Note: This check the nuclide to determine if the nuclide has the GammaData
     * if the nuclide's GammaData is zero, it attempts to retrieve the value from the nuclide
     * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
     * given nuclide will be returned.
     */
    static int getGammaData(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the ThermalNeutronData
     * @param nuclide - the nuclide for which to obtain the ThermalNeutronData
     * @param library - the library that backs up the nuclide
     * @return int - the ThermalNeutronData
     * Note: This check the nuclide to determine if the nuclide has the ThermalNeutronData
     * if the nuclide's ThermalNeutronData is zero, it attempts to retrieve the value from the nuclide
     * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
     * given nuclide will be returned.
     */
    static int getThermalNeutronData(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the FastNeutronData
     * @param nuclide - the nuclide for which to obtain the FastNeutronData
     * @param library - the library that backs up the nuclide
     * @return int - the FastNeutronData
     * Note: This check the nuclide to determine if the nuclide has the FastNeutronData
     * if the nuclide's FastNeutronData is zero, it attempts to retrieve the value from the nuclide
     * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
     * given nuclide will be returned.
     */
    static int getFastNeutronData(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the PotentialScatter
     * @param nuclide - the nuclide for which to obtain the PotentialScatter
     * @param library - the library that backs up the nuclide
     * @return float - the PotentialScatter
     * Note: This check the nuclide to determine if the nuclide has the PotentialScatter
     * if the nuclide's PotentialScatter is zero, it attempts to retrieve the value from the nuclide
     * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
     * given nuclide will be returned.
     */
    static float getPotentialScatter(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the MaxLengthScatterArray
     * @param nuclide - the nuclide for which to obtain the MaxLengthScatterArray
     * @param library - the library that backs up the nuclide
     * @return int - the MaxLengthScatterArray
     * Note: This check the nuclide to determine if the nuclide has the MaxLengthScatterArray
     * if the nuclide's MaxLengthScatterArray is zero, it attempts to retrieve the value from the nuclide
     * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
     * given nuclide will be returned.
     */
    static int getMaxLengthScatterArray(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the EnergyReleasePerCapture
     * @param nuclide - the nuclide for which to obtain the EnergyReleasePerCapture
     * @param library - the library that backs up the nuclide
     * @return float - the EnergyReleasePerCapture
     * Note: This check the nuclide to determine if the nuclide has the EnergyReleasePerCapture
     * if the nuclide's EnergyReleasePerCapture is zero, it attempts to retrieve the value from the nuclide
     * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
     * given nuclide will be returned.
     */
    static float getEnergyReleasePerCapture(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the PowerPerFission
     * @param nuclide - the nuclide for which to obtain the PowerPerFission
     * @param library - the library that backs up the nuclide
     * @return float - the PowerPerFission
     * Note: This check the nuclide to determine if the nuclide has the PowerPerFission
     * if the nuclide's PowerPerFission is zero, it attempts to retrieve the value from the nuclide
     * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
     * given nuclide will be returned.
     */
    static float getPowerPerFission(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the ZA
     * @param nuclide - the nuclide for which to obtain the ZA
     * @param library - the library that backs up the nuclide
     * @return int - the ZA
     * Note: This check the nuclide to determine if the nuclide has the ZA
     * if the nuclide's ZA is zero, it attempts to retrieve the value from the nuclide
     * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
     * given nuclide will be returned.
     */
    static int getZA(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the AtomicMass
     * @param nuclide - the nuclide for which to obtain the AtomicMass
     * @param library - the library that backs up the nuclide
     * @return float - the AtomicMass
     * Note: This check the nuclide to determine if the nuclide has the AtomicMass
     * if the nuclide's AtomicMass is zero, it attempts to retrieve the value from the nuclide
     * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
     * given nuclide will be returned.
     */
    static float getAtomicMass(LibraryNuclide * nuclide, AmpxLibrary * library);

    /**
     * @brief Obtain the NumEnergiesUnresolvedResonance
     * @param nuclide - the nuclide for which to obtain the NumEnergiesUnresolvedResonance
     * @param library - the library that backs up the nuclide
     * @return int - the NumEnergiesUnresolvedResonance
     * Note: This check the nuclide to determine if the nuclide has the NumEnergiesUnresolvedResonance
     * if the nuclide's NumEnergiesUnresolvedResonance is zero, it attempts to retrieve the value from the nuclide
     * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
     * given nuclide will be returned.
     */
    static int getNumEnergiesUnresolvedResonance(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the NumResolvedResonance
     * @param nuclide - the nuclide for which to obtain the NumResolvedResonance
     * @param library - the library that backs up the nuclide
     * @return int - the NumResolvedResonance
     * Note: This check the nuclide to determine if the nuclide has the NumResolvedResonance
     * if the nuclide's NumResolvedResonance is zero, it attempts to retrieve the value from the nuclide
     * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
     * given nuclide will be returned.
     */
    static int getNumResolvedResonance(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the MaxNumBondGrp
     * @param nuclide - the nuclide for which to obtain the MaxNumBondGrp
     * @param library - the library that backs up the nuclide
     * @return int - the MaxNumBondGrp
     * Note: This check the nuclide to determine if the nuclide has the MaxNumBondGrp
     * if the nuclide's MaxNumBondGrp is zero, it attempts to retrieve the value from the nuclide
     * on the ampxlibrary. If the nuclide does not exist on the library, the value (probably zero) from the
     * given nuclide will be returned.
     */
    static int getMaxNumBondGrp(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the global bondarenko data for the given nuclide
     * @param nuclide - the nuclide from which to obtain the global bondarenko data
     * @param library - the library that backs up the nuclide
     * @param readOnly - indicator of whether or not the caller intends to write/ modify the data
     * @return BondarenkoGlobal * - BondarenkoGlobal data, or null if none was found on either the nuclide or the library
     * Note: If the data is not found on the nuclide, but is found on the library and the readOnly=false, then the global data
     * will be copied onto the nuclide for modification and future retrieval.
     */
    static BondarenkoGlobal* getBondarenkoGlobalData(LibraryNuclide* nuclide, AmpxLibrary * library, bool readOnly);
    /**
     * @brief Obtain the list of 1d mts that are available for the given nuclide
     * @param nuclide - the nuclide for which to obtain the list of mts
     * @param library - the ampx library that backs up the nuclide
     * @return QList<int> * - a newly allocated list of nuclide ids (THIS WILL NEED TO BE DELETED BY YOU)
     */
    static QList<int> * getBondarenkoDataMts(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the neutron bondarenko data 1d with the given mt from either the nuclide or the ampx library
     * This method obtains the bondarenko data 1d with the given mt from the nuclide or the library. The method
     * check the nuclide first, and if the nuclide does not contain the given mt, it attempts to retrieve the bondarenko data
     * from the library.
     *
     * If the bondarenko data is retrieved from the library and readOnly==false, the bondarenko data is copied onto
     * the nuclide, such that any writes will not effect the original library bondarenko data.
     * @param nuclide - the nuclide from which the bondarenko data will come from or be placed on if not present
     * @param library - the library from which the bondarenko data will be obtained if not on the nuclide
     * @param mt - the type of bondarenko data to be obtained
     * @param readOnly - if the bondarenko data is obtained from the library, this indicates if the library needs to be copied onto the nuclide
     * @return BondarenkoData* - the desired bondarenko datas obtained from the nuclide or the library, NULL if none were available
     */
    static BondarenkoData* getBondarenkoDataByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt, bool readOnly);
    /**
     * @brief Obtain the list of 1d mts that are available for the given nuclide
     * @param nuclide - the nuclide for which to obtain the list of mts
     * @param library - the ampx library that backs up the nuclide
     * @return QList<int> * - a newly allocated list of nuclide ids (THIS WILL NEED TO BE DELETED BY YOU)
     */
    static QList<int> * getNeutron1dXSMts(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the neutron cross section 1d with the given mt from either the nuclide or the ampx library
     * This method obtains the cross section 1d with the given mt from the nuclide or the library. The method
     * check the nuclide first, and if the nuclide does not contain the given mt, it attempts to retrieve the cross section
     * from the library.
     *
     * If the cross section is retrieved from the library and readOnly==false, the cross section is copied onto
     * the nuclide, such that any writes will not effect the original library cross section.
     * @param nuclide - the nuclide from which the cross section will come from or be placed on if not present
     * @param library - the library from which the cross section will be obtained if not on the nuclide
     * @param mt - the type of cross section to be obtained
     * @param readOnly - if the cross section is obtained from the library, this indicates if the library needs to be copied onto the nuclide
     * @return CrossSection1d* - the desired cross sections obtained from the nuclide or the library, NULL if none were available
     */
    static CrossSection1d * getNeutron1dXSByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt, bool readOnly);
    /**
     * @brief Obtain the list of 2d mts that are available for the given nuclide
     * @param nuclide - the nuclide for which to obtain the list of mts
     * @param library - the ampx library that backs up the nuclide
     * @return QList<int> * - a newly allocated list of nuclide ids (THIS WILL NEED TO BE DELETED BY YOU)
     */
    static QList<int> * getNeutron2dXSMts(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the neutron cross section 2d with the given mt from either the nuclide or the ampx library
     * This method obtains the cross section 2d with the given mt from the nuclide or the library. The method
     * check the nuclide first, and if the nuclide does not contain the given mt, it attempts to retrieve the cross section
     * from the library.
     *
     * If the cross section is retrieved from the library and readOnly==false, the cross section is copied onto
     * the nuclide, such that any writes will not effect the original library cross section.
     * @param nuclide - the nuclide from which the cross section will come from or be placed on if not present
     * @param library - the library from which the cross section will be obtained if not on the nuclide
     * @param mt - the type of cross section to be obtained
     * @param readOnly - if the cross section is obtained from the library, this indicates if the library needs to be copied onto the nuclide
     * @param CrossSection2d - the desired cross sections obtained from the nuclide or the library, NULL if none were available
     */
    static CrossSection2d * getNeutron2dXSByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt, bool readOnly);

    /**
     * @brief Obtain the list of 1d mts that are available for the given nuclide
     * @param nuclide - the nuclide for which to obtain the list of mts
     * @param library - the ampx library that backs up the nuclide
     * @return QList<int> * - a newly allocated list of nuclide ids (THIS WILL NEED TO BE DELETED BY YOU)
     */
    static QList<int> * getGamma1dXSMts(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the neutron cross section 1d with the given mt from either the nuclide or the ampx library
     * This method obtains the cross section 1d with the given mt from the nuclide or the library. The method
     * check the nuclide first, and if the nuclide does not contain the given mt, it attempts to retrieve the cross section
     * from the library.
     *
     * If the cross section is retrieved from the library and readOnly==false, the cross section is copied onto
     * the nuclide, such that any writes will not effect the original library cross section.
     * @param nuclide - the nuclide from which the cross section will come from or be placed on if not present
     * @param library - the library from which the cross section will be obtained if not on the nuclide
     * @param mt - the type of cross section to be obtained
     * @param readOnly - if the cross section is obtained from the library, this indicates if the library needs to be copied onto the nuclide
     * @return CrossSection1d* - the desired cross sections obtained from the nuclide or the library, NULL if none were available
     */
    static CrossSection1d * getGamma1dXSByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt, bool readOnly);
    /**
     * @brief Obtain the list of 2d mts that are available for the given nuclide
     * @param nuclide - the nuclide for which to obtain the list of mts
     * @param library - the ampx library that backs up the nuclide
     * @return QList<int> * - a newly allocated list of nuclide ids (THIS WILL NEED TO BE DELETED BY YOU)
     */
    static QList<int> * getGamma2dXSMts(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the neutron cross section 2d with the given mt from either the nuclide or the ampx library
     * This method obtains the cross section 2d with the given mt from the nuclide or the library. The method
     * check the nuclide first, and if the nuclide does not contain the given mt, it attempts to retrieve the cross section
     * from the library.
     *
     * If the cross section is retrieved from the library and readOnly==false, the cross section is copied onto
     * the nuclide, such that any writes will not effect the original library cross section.
     * @param nuclide - the nuclide from which the cross section will come from or be placed on if not present
     * @param library - the library from which the cross section will be obtained if not on the nuclide
     * @param mt - the type of cross section to be obtained
     * @param readOnly - if the cross section is obtained from the library, this indicates if the library needs to be copied onto the nuclide
     * @param CrossSection2d - the desired cross sections obtained from the nuclide or the library, NULL if none were available
     */
    static CrossSection2d * getGamma2dXSByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt, bool readOnly);

    /**
     * @brief Obtain the list of Prod mts that are available for the given nuclide
     * @param nuclide - the nuclide for which to obtain the list of mts
     * @param library - the ampx library that backs up the nuclide
     * @return QList<int> * - a newly allocated list of nuclide ids (THIS WILL NEED TO BE DELETED BY YOU)
     */
    static QList<int> * getGammaProdXSMts(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Obtain the neutron cross section Prod with the given mt from either the nuclide or the ampx library
     * This method obtains the cross section Prod with the given mt from the nuclide or the library. The method
     * check the nuclide first, and if the nuclide does not contain the given mt, it attempts to retrieve the cross section
     * from the library.
     *
     * If the cross section is retrieved from the library and readOnly==false, the cross section is copied onto
     * the nuclide, such that any writes will not effect the original library cross section.
     * @param nuclide - the nuclide from which the cross section will come from or be placed on if not present
     * @param library - the library from which the cross section will be obtained if not on the nuclide
     * @param mt - the type of cross section to be obtained
     * @param readOnly - if the cross section is obtained from the library, this indicates if the library needs to be copied onto the nuclide
     * @param CrossSectionProd - the desired cross sections obtained from the nuclide or the library, NULL if none were available
     */
    static CrossSection2d * getGammaProdXSByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt, bool readOnly);
    /**
     * @brief Obtain the maximum legendre order for any neutron2d cross section contained in the given nuclide
     * This will be the max legendre order of any neutron2d cross section on the nuclide or the library
     * @param nuclide - the nuclide from which the cross section will come from or be placed on if not present
     * @param library - the library from which the cross section will be obtained if not on the nuclide
     * @return int - the max legendre order for any neutron2d cross section related to the given nuclide
     */
    static int getMaxNeutron2dLegendreOrder(LibraryNuclide * nuclide, AmpxLibrary * library );
    /**
     * @brief Obtain the maximum legendre order for any gamma2d cross section contained in the given nuclide
     * This will be the max legendre order of any gamma2d cross section on the nuclide or the library
     * @param nuclide - the nuclide from which the cross section will come from or be placed on if not present
     * @param library - the library from which the cross section will be obtained if not on the nuclide
     * @return int - the max legendre order for any gamma2d cross section related to the given nuclide
     */
    static int getMaxGamma2dLegendreOrder(LibraryNuclide * nuclide, AmpxLibrary * library );

    /**
     * @brief Determine if this nuclide contains any Neutron1d data
     * @param nuclide - the nuclide from which to determine the presence of Neutron1d data
     * @param library - the ampx library that backs up this nuclide.
     * @return bool - true, if Neutron1d data is present on either the nuclide or the library nuclide
     *  that backs up the given nuclide
     */
    static bool containsNeutron1dData(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Determine if this nuclide contains any Neutron2d data
     * @param nuclide - the nuclide from which to determine the presence of Neutron2d data
     * @param library - the ampx library that backs up this nuclide.
     * @return bool - true, if Neutron2d data is present on either the nuclide or the library nuclide
     *  that backs up the given nuclide
     */
    static bool containsNeutron2dData(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Determine if this nuclide contains any Gamma1d data
     * @param nuclide - the nuclide from which to determine the presence of Gamma1d data
     * @param library - the ampx library that backs up this nuclide.
     * @return bool - true, if Gamma1d data is present on either the nuclide or the library nuclide
     *  that backs up the given nuclide
     */
    static bool containsGamma1dData(LibraryNuclide * nuclide, AmpxLibrary * library);
    /**
     * @brief Determine if this nuclide contains any Gamma2d data
     * @param nuclide - the nuclide from which to determine the presence of Gamma2d data
     * @param library - the ampx library that backs up this nuclide.
     * @return bool - true, if Gamma2d data is present on either the nuclide or the library nuclide
     *  that backs up the given nuclide
     */
    static bool containsGamma2dData(LibraryNuclide * nuclide, AmpxLibrary * library);

    /**
     * @brief Determine if this nuclide contains any GammaProd data
     * @param nuclide - the nuclide from which to determine the presence of GammaProd data
     * @param library - the ampx library that backs up this nuclide.
     * @return bool - true, if GammaProd data is present on either the nuclide or the library nuclide
     *  that backs up the given nuclide
     */
    static bool containsGammaProdData(LibraryNuclide * nuclide, AmpxLibrary * library);

    /**
     * @brief Determine if this nuclide contains a given  Neutron1d data
     * @param nuclide - the nuclide from which to determine the presence of Neutron1d data
     * @param library - the ampx library that backs up this nuclide.
     * @return bool - true, if Neutron1d data is present on either the nuclide or the library nuclide
     *  that backs up the given nuclide
     */
    static bool containsNeutron1dDataByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt);
    /**
     * @brief Determine if this nuclide contains a given  Neutron2d data
     * @param nuclide - the nuclide from which to determine the presence of Neutron2d data
     * @param library - the ampx library that backs up this nuclide.
     * @return bool - true, if Neutron2d data is present on either the nuclide or the library nuclide
     *  that backs up the given nuclide
     */
    static bool containsNeutron2dDataByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt);
    /**
     * @brief Determine if this nuclide contains a given  Gamma1d data
     * @param nuclide - the nuclide from which to determine the presence of Gamma1d data
     * @param library - the ampx library that backs up this nuclide.
     * @return bool - true, if Gamma1d data is present on either the nuclide or the library nuclide
     *  that backs up the given nuclide
     */
    static bool containsGamma1dDataByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt);
    /**
     * @brief Determine if this nuclide contains a given  Gamma2d data
     * @param nuclide - the nuclide from which to determine the presence of Gamma2d data
     * @param library - the ampx library that backs up this nuclide.
     * @return bool - true, if Gamma2d data is present on either the nuclide or the library nuclide
     *  that backs up the given nuclide
     */
    static bool containsGamma2dDataByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt);

    /**
     * @brief Determine if this nuclide contains a given  GammaProd data
     * @param nuclide - the nuclide from which to determine the presence of GammaProd data
     * @param library - the ampx library that backs up this nuclide.
     * @return bool - true, if GammaProd data is present on either the nuclide or the library nuclide
     *  that backs up the given nuclide
     */
    static bool containsGammaProdDataByMt(LibraryNuclide * nuclide, AmpxLibrary * library, int mt);
};

#endif	/* AMPXDATAHELPER_H */
