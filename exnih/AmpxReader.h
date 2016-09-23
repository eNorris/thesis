#ifndef AMPXREADER_H
#define AMPXREADER_H
#include <QList>
#include <iostream>
#include <fstream>
#include <QString>
#include "AmpxLibrary.h"

using namespace std;

class LibraryHeader;
class LibraryEnergyBounds;
class LibraryNuclide;
class NuclideResonance;
class BondarenkoGlobal;
class BondarenkoData;
class BondarenkoInfiniteDiluted;
class BondarenkoFactors;
class CrossSection1d;
class CrossSection2d;
class ScatterMatrix;

class AmpxReader
{
protected:
    bool swapEndian;
    LibraryHeader * readLibraryHeader(fstream & is,
                               int * pos=NULL,
                               bool verbose=false);
    LibraryEnergyBounds * readEnergyBounds(fstream & is,
                                            int numGroups,
                                            int * pos=NULL,
                                            bool verbose=false);

    LibraryNuclide* readLibraryNuclide(fstream & is,
                                int * pos=NULL,
                                bool verbose=false);
    BondarenkoGlobal * readBondarenkoGlobal(fstream & is,
                                            int numSig0,
                                            int numTemps,
                                            int * pos=NULL,
                                            bool verbose=false);

    /// reads a nuclide's bondarenko data sets
    /// and returns a list of objects
    /// @param fstream &is - The file from which to read the data
    /// @param int numBondSets - The number of bondarenko data sets to read
    /// @param int numBondSig0- The number of bondarenko Sig0 data sets to read
    /// @param int numBondTemps - The number of bondarenko Temp data sets to read
    /// @param int * pos=NULL - The optional position from which to read
    /// @param bool verbose - be verbose about activity
    QList<BondarenkoData*> readBondarenkoData(fstream & is,
                        int numBondSets,
                        int numBondSig0,
                        int numBondTemps,
                        int * pos=NULL, bool verbose=false);

    BondarenkoInfiniteDiluted * readBondarenkoInfiniteDiluted(fstream & is,
                                                            int startGrp,
                                                            int endGrp,
                                                            int * pos=NULL,
                                                            bool verbose=false);

    BondarenkoFactors * readBondarenkoFactors(fstream & is,
                                                int numSig,
                                                int numTemp,
                                                int startGrp,
                                                int endGrp,
                                                int * pos=NULL,
                                                bool verbose=false);

    QList<CrossSection1d *> readCrossSection1d(fstream & is,
                                                int num1d,
                                                int numGrps,
                                                int * pos=NULL,
                                                bool verbose=false);

    QList<CrossSection2d*> readCrossSection2d(fstream & is,
                                                int num2d,
                                                int type,
                                                int * pos=NULL,
                                                bool verbose=false);

    ScatterMatrix * readScatterMatrix(fstream & is,
                                        int length,
                                        int type,
                                        int * pos=NULL,
                                        bool verbose=false);
    QList<ScatterMatrix *> readScatterMatrices(fstream & is,
                                        int numTemps,
                                        int maxLegendreOrder,
                                        int length,
                                        int type,
                                        int * pos=NULL,
                                        bool verbose=false);

    NuclideResonance * readNuclideResonance(fstream & is,
                                            int numResolved,
                                            int numUnresolved,
                                            int * pos=NULL,
                                            bool verbose=false);

    void initialize();
    bool readLibraryHeader(fstream & file, AmpxLibrary & library);
    bool readEnergyBounds(fstream & file, AmpxLibrary & library);
    virtual bool readNuclideInfo(fstream & file, AmpxLibrary & library);
    virtual bool readNuclideData(fstream & file, AmpxLibrary & library);
    QString readError;
    bool printErrors;
    bool verboseOutput;
public:
    AmpxReader();

    int readHeaderInfo(fstream * file, AmpxLibrary & library, bool printErrors = true, bool verbose=false);

    int read(fstream * file, AmpxLibrary & library, bool printErrors = true, bool verbose=false);
    QString getReadError(){return readError;}
    /// obtain the footprint size of a library energy bounds
    /// given the number of groups
    /// @param int numGrps - The number of groups for the energy bounds
    static int footprintLibraryEnergyBounds(int numGrps);
    static int footprintLibraryNuclide();
    static int footprintLibraryHeader();
    /// obtain the size of a NuclideResonance objects footprint on disk
    /// given the number of resolved and unresolved
    /// @param int numResolved - Number of 6-parameter sets of resolved resonance data
    /// @param int numUnresolved - Number of energies at which to evaluate unresolved values
    /// @return int - the size of the given NuclideResonance
    static int footprintNuclideResonance(int numResolved, int numUnresolved);

    /// obtain the size of a ScatterMatrix objects footprint on disk
    /// @param int length - the max length
    /// @param int type - the type of cross section
    ///                  This should be AMPXLIB_NEUTRON2D_DATA,
    ///                  AMPXLIB_GAMMA2D_DATA, or AMPXLIB_GAMMAPRODUCTION_DATA
    /// @return int - the size of the given ScatterMatrix
    static int footprintScatterMatrix(int length, int type);
    static int footprintCrossSection2d();

    /// obtain the size of a crossection1d objects footprint on disk
    /// given the number of sig0 and temps
    /// @param int numGrps - the first grp for which these infinite diluted values represent
    /// @return int - the size of the given CrossSection1d
    static int footprintCrossSection1d(int numGrps);

    /// obtain the size of a bondarenkofactors objects footprint on disk
    /// given the number of sig0 and temps
    /// @param int numSig - the number of sig0
    /// @param int numTemp - the number of temperatures
    /// @param int startGrp - the first grp for which these infinite diluted values represent
    /// @param int endGrp - the last grp for which these infinite diluted values represent
    /// @return int - the size of the given BondarenkoFactors
    static int footprintBondarenkoFactors(int numSig, int numTemp, int startGrp, int endGrp);

    /// obtain the size of a bondarenkoinfinitediluted objects footprint on disk
    /// given the number of sig0 and temps
    /// @param int startGrp - the first grp for which these infinite diluted values represent
    /// @param int endGrp - the last grp for which these infinite diluted values represent
    /// @return int - the size of the given BondarenkoInfiniteDiluted
    static int footprintBondarenkoInfiniteDiluted(int startGrp, int endGrp);

    /// obtain the size of a bondarenkoglobal objects footprint on disk
    /// given the number of sig0 and temps
    /// @param int numSig0 - the number of sig0 values expected
    /// @param int numTemp - the number of temperature values expected
    /// @return int - the size of the given BondarenkoGlobal
    static int footprintBondarenkoGlobal(int numSig0, int numTemp);
    static int footprintBondarenkoData();
};
#endif
