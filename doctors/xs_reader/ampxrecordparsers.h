#ifndef AMPXRECORDPARSERS_H
#define AMPXRECORDPARSERS_H

#include <vector>
#include <fstream>
#include <algorithm>
#include <string>

using std::ifstream;

template <class T>
void endswap(T *objp)
{
    unsigned char *memp = reinterpret_cast<unsigned char*>(objp);
    std::reverse(memp, memp + sizeof(T));
}

void nextBinInt(ifstream &binfile, int &v);
int nextBinInt(ifstream &binfile);
void nextBinFloat(ifstream &binfile, float &v);
void nextBinString(ifstream &binfile, unsigned int words32bit, char array[]);

void nextBinFloats(ifstream &binfile, float &v, int count);

class AmpxRecordParserBase
{
protected:
    bool built;
    int recordSize;
    int recordSizePrev;

public:
    AmpxRecordParserBase() : built(false), recordSize(-1), recordSizePrev(-1) {}
    ~AmpxRecordParserBase() {}

    void parse(ifstream &binfile);
    void parseHeader(ifstream &binfile);
};


// ---------------------------------------------------- //
// ---------------------- Type 1 ---------------------- //
// ---------------------------------------------------- //
class AmpxRecordParserType1 : public AmpxRecordParserBase
{
protected:
    /** An identification number for the library */
    int idtape;

    /** The number of sets of data on the library */
    int nnuc;

    /** The number of neutron energy groups on the library */
    int igm;

    /** The first thermal neutron group on the library (first group that _receives_ an upscatter source) */
    int iftg;

    /** Master Library version type (2 for NITAWL-II resonance processing) */
    int msn;

    /** The number of gamma ray energy groups on the library */
    int ipm;

    /** Zero */
    int i1;

    /** (1/0 = yes/no) A trigger that specifies that this library was produced by weighting a working library in the XSDRNPM module */
    int i2;

    /** Zero */
    int i3;

    /** Zero */
    int i4;

    /** 100 words of text describing the cross-section library */
    char title[100*4];

public:
    AmpxRecordParserType1();
    ~AmpxRecordParserType1();

    int getTapeId() const { return idtape; }
    int getNumberNuclides() const { return nnuc; }
    int getGroupCountNeutron() const { return igm; }
    int getGroupCountGamma() const { return ipm; }
    int getFirstThermalGroup() const { return iftg; }
    int getMasterLibVersion() const { return msn; }
    int getXsdrnpmFlag() const { return i2; }

    bool parse(ifstream &binfile);
    bool validate();
};


// ---------------------------------------------------- //
// ---------------------- Type 2 ---------------------- //
// ---------------------------------------------------- //
class AmpxRecordParserType2 : public AmpxRecordParserBase
{
protected:
    /** Energy boundaries between bins in eV */
    std::vector<float> energy;

    /** Lethargy boundaries between bins, typically 0 at 10 MeV */
    std::vector<float> lethargy;

public:
    AmpxRecordParserType2() : AmpxRecordParserBase() {}
    ~AmpxRecordParserType2() {}

    bool parse(ifstream &binfile, int energyBins);

    const std::vector<float> &getEnergy() const { return energy; }
    const std::vector<float> &getLethargy() const { return lethargy; }
};


// ---------------------------------------------------- //
// ---------------------- Type 3 ---------------------- //
// ---------------------------------------------------- //
/** Cross-Section Set Directory Record */
class AmpxRecordParserType3 : public AmpxRecordParserBase
{
protected:
    /** 18 words of text describing the set */
    char text[18*4];  // 1-18

    /** Identifier of the set */
    int id;           // 19

    /** Number of 6-parameter sets of resolved resonance data */
    int nres;         // 20

    /** Number of energies at which to evaluate unresolved values */
    int nunr;         // 21

    /** Number o fneutron processes fo rwhich group-averaged values are given (temp independent) */
    int navg;         // 22

    /** Number of processes with scattering arrays */
    int n2d;          // 23

    /** Zero */
    int w24;          // 24

    /** Number of gamma processes for which group-averaged values are given */
    int gavg;         // 25

    /** Number of gamma processes with scattering arrays */
    int g2d;          // 26

    /** Number of neutron-to-gamma processes */
    int ngct;         // 27

    /** (Maximum order of scattering)*32768 + (total number of separate scattering arrays for this set) */
    int ord;          // 28

    /** A - neutron equivalent mass number */
    float mass;       // 29

    /** ZA - 1000*Z + A */
    float za;         // 30

    /** Zero */
    int w31;          // 31

    /** Zero */
    int w32;          // 32

    /** Zero */
    int w33;          // 33

    /** Power per fission in wat-sec/fission */
    float pwr;        // 34

    /** Energy release per capture in wat-sec/capture */
    float ec;         // 35

    /** Maximum length of any scattering array in the set */
    int maxs;         // 36

    /** Number of sets of Bondarenko data */
    int nbond;        // 37

    /** Number of sigma_0 values in Bondarenko data */
    int nsig;         // 38

    /** Number of temperature values in Bondarenko data */
    int nt;           // 39

    /** Maximum number of groups in Bondarenko data */
    int nbondg;       // 40

    /** Zero */
    int w41;          // 41

    /** Zero */
    int w42;          // 42

    /** sigma_p - potential scattering cross section */
    float sigp;       // 43

    /** Zero */
    int w44;          // 44

    /** ENDF MAT for fast neutron data */
    int endffn;       // 45

    /** ENDF MAT for thermal neutron data */
    int endftn;       // 46

    /** ENDF MAT for gamma data */
    int endfg;        // 47

    /** ENDF MAT for gamma production data */
    int endfgp;       // 48

    /** Source: 0 = ENDF */
    char sym[1*4];    // 49

    /** Number of records in this set */
    int nrec;         // 50

public:
    AmpxRecordParserType3();
    ~AmpxRecordParserType3();

    bool parse(ifstream &binfile);

    const char *getText() const { return text; }              // 1-18
    int getId() const { return id; }                          // 19
    int getResolvedGroups() const { return nres; }            // 20
    int getUnresolvedGroups() const { return nunr; }          // 21
    int getAveragedNeutronProcCount() const { return navg; }  // 22
    int getScatterNeutronProcCount() const { return n2d; }    // 23
    int getAveragedGammaProcCount() const { return gavg; }    // 25
    int getScatterGammaProcCount() const { return g2d; }      // 26
    int getNeutronToGammaProcCount() const { return ngct; }   // 27
    int getMaxScatterSize() const { return maxs; }            // 36
    int getBondarenkoCount() const { return nbond; }          // 37
    int getBondarenkoSig0Count() const { return nsig; }       // 38
    int getBondarenkoTempCount() const { return nt; }         // 39
    int getBondarenkoGroupMax() const { return nbondg; }      // 40
    int getRecordCount() const { return nrec; }               // 50

    std::string getTextAsStdString() const { return std::string(text); }
};


// ---------------------------------------------------- //
// ---------------------- Type 4 ---------------------- //
// ---------------------------------------------------- //
/** Depricated */
class AmpxRecordParserType4 : public AmpxRecordParserBase
{
protected:

public:
    AmpxRecordParserType4() : AmpxRecordParserBase() {}
    ~AmpxRecordParserType4() {}

    bool parse(ifstream &binfile, int nres, int nunr);
};


// ---------------------------------------------------- //
// ---------------------- Type 5 ---------------------- //
// ---------------------------------------------------- //
class AmpxRecordParserType5 : public AmpxRecordParserBase
{
protected:
    /** sigma_0 Bondarenko factors */
    std::vector<float> sig0;

    /** temperatures for Bondarenko factors */
    std::vector<float> temp;

    /** Lower energy for which Bondarenko factors apply */
    float elo;

    /** Upper energy for which Bondarenko factors apply */
    float ehi;

public:
    AmpxRecordParserType5();
    ~AmpxRecordParserType5();

    bool parse(ifstream &binfile, int nsig0, int nt);

    int getTempCount() const { return temp.size(); }
    int getSig0Count() const { return sig0.size(); }
};


// ---------------------------------------------------- //
// ---------------------- Type 6 ---------------------- //
// ---------------------------------------------------- //
/** Directory for Bondarenko block */
class AmpxRecordParserType6 : public AmpxRecordParserBase
{
protected:
    /** MT process identifier */
    std::vector<int> mt;

    /** Number of the first energy group for which parameters are given */
    std::vector<int> nf;

    /** Number of the last energy group for which parameters are given */
    std::vector<int> nl;

    /** Specifies the lower group of homogeneous or heterogeneous f-factor*/
    std::vector<int> order;

    /** Specifies the upper group of homogeneous or heterogeneous f-factor*/
    std::vector<int> ioff;

    /** Unused - filled with zeros */
    std::vector<int> nz;

public:
    AmpxRecordParserType6();
    ~AmpxRecordParserType6();

    bool parse(ifstream &binfile, int nbond);
    const std::vector<int> &getMt() const { return mt; }
    const std::vector<int> &getNf() const { return nf; }
    const std::vector<int> &getNl() const { return nl; }
    const std::vector<int> &getOrder() const { return order; }
    const std::vector<int> &getIoff() const { return ioff; }
    const std::vector<int> &getNz() const { return nz; }
};


// ---------------------------------------------------- //
// ---------------------- Type 7 ---------------------- //
// ---------------------------------------------------- //
/** Infinite Dilution Values for Bondarenko Data */
class AmpxRecordParserType7 : public AmpxRecordParserBase
{
protected:
    /** sigma_inf */
    std::vector<float> sigInf;

public:
    AmpxRecordParserType7();
    ~AmpxRecordParserType7();

    bool parse(ifstream &binfile, int firstGroup, int lastGroup);
};


// ---------------------------------------------------- //
// ---------------------- Type 8 ---------------------- //
// ---------------------------------------------------- //
/** Bondarenko Factors */
class AmpxRecordParserType8 : public AmpxRecordParserBase
{
protected:
    int nf;
    int nl;
    int nt;
    int nsig0;

    // Index into a flattened array of [num energy * num temp * num sig0]
    std::vector<float> bond;

public:
    AmpxRecordParserType8();
    ~AmpxRecordParserType8();

    bool parse(ifstream &binfile, int nf, int nl, int num_t, int num_sig0);

    int getFirstGroup() const { return nf; }
    int getLastGroup() const { return nl; }
    int getNumberTemps() const { return nt; }
    int getNumberSig0() const { return nsig0; }
    float getBondarenkoSig0(int eGroup, int tempIndx, int sig0Indx) const;
};


// ---------------------------------------------------- //
// ---------------------- Type 9 ---------------------- //
// ---------------------------------------------------- //
/** Temperature-Independent Average Cross Sections */
class AmpxRecordParserType9 : public AmpxRecordParserBase
{
protected:
    /** MT Process identifiers */
    std::vector<float> mt;

    /** Cross section */
    std::vector<float> sigma;

    int groupCount;

public:
    AmpxRecordParserType9();
    ~AmpxRecordParserType9();

    bool parse(ifstream &binfile, int mtCount, int groups);

    int getMtCount() const { return mt.size(); }
    const std::vector<float> &getMtList() const { return mt; }
    int getMtIndex(const int mtId) const;
    const std::vector<float> &getSigma() const { return sigma; }
    std::vector<float> getSigmaMt(int mtIndex) const;
    float getSigma(const int mtIndex, const int eIndex) const { return sigma[mt.size() * mtIndex + eIndex]; }
    std::vector<float> getSigma(const int mtIndex) const;
};


// ---------------------------------------------------- //
// ---------------------- Type 10 --------------------- //
// ---------------------------------------------------- //
/** Scattering Matrix Directory */
class AmpxRecordParserType10 : public AmpxRecordParserBase
{
protected:
    /** MT Process identifier */
    std::vector<int> mt;

    /** Maximum length of any of the scattering matrices for the process */
    std::vector<int> l;

    /** The order of Legendre fit to the scattering data */
    std::vector<int> nl;

    /** Neutron-Neutron data: Number of temperatures at which scattering matrices are given. May be 0 if only one temperature is present
     * Gamma-Production data: Zero if the data are in yield units, unity if cross section units
     * Gamma-Gamma data: Zero */
    std::vector<int> nt;

public:
    AmpxRecordParserType10();
    ~AmpxRecordParserType10();

    bool parse(ifstream &binfile, int nProcesses);

    const std::vector<int> &getMtList() const { return mt; }
    const std::vector<int> &getLList() const { return l; }
    const std::vector<int> &getNlList() const { return nl; }
    const std::vector<int> &getTemp() const { return nt; }

    int getMtIndex(const int mtId) const;
    int getNlIndex(const int nlId) const;
};


// ---------------------------------------------------- //
// ---------------------- Type 11 --------------------- //
// ---------------------------------------------------- //
/** Scattering Matrix Temperatures */
class AmpxRecordParserType11 : public AmpxRecordParserBase
{
protected:
    /** Temperature (eV) of scattering matrices, used only for neutron-neutron data when nt > 10 (nt from Record Type 10)*/
    std::vector<float> temps;

public:
    AmpxRecordParserType11();
    ~AmpxRecordParserType11();

    bool parse(ifstream &binfile, int num_temps);
};


// ---------------------------------------------------- //
// ---------------------- Type 12 --------------------- //
// ---------------------------------------------------- //
/** Scattering Matrix */
class AmpxRecordParserType12 : public AmpxRecordParserBase
{
protected:
    int size;
    int maxSize;
    int groups;

    /** Sink Group Number, III
     * First group number, JJJ, which scatters into this group
     * Last group number, KKK, which scatters into this gorup
     * Magic = 1000000*JJJ + 1000*KKK + III = JJJKKKIII
     *
     * MW for group III
     * sigma(KKK -> III)
     * sigma(KKK-1 -> III)
     * ...
     * sigma(JJJ -> III) */
    std::vector<int> magics;

    //float *xsData;
    /** Scatter matrix, formatted as sigma(src -> sink) = xsData[(src-1)*Energy_Groups + sink - 1] */
    std::vector<float> xsData;

public:
    AmpxRecordParserType12();
    ~AmpxRecordParserType12();

    bool parse(ifstream &binfile, int maxlen, bool selfDefining, int nGroups);
    const std::vector<float> &getXsVector() const;
    float getXs(int srcEGrpIndex, int sinkEGrpIndex) const;
};

#endif // AMPXRECORDPARSERS_H
