#ifndef CONFIG_H
#define CONFIG_H

#include "globals.h"

#include <string>
#include <vector>

class Config
{
public:
    Config();
    ~Config();

    void loadDefaults();

    std::string resultsBasename;

    std::string spectrumFile;
    float sourceFanAngle;
    float sourceConeAngle;
    float sourceTopGap;  // [cm]
    float sourceBottomGap; // [cm]
    float sourceLeftGap;
    float sourceRightGap;
    float sourceFrontGap;
    float sourceBackGap;
    float colXLen;  // [cm]  Collimator x length
    float colYLen;
    float colZLen;

    std::vector<float> sourceX;
    std::vector<float> sourceY;
    std::vector<float> sourceZ;
    std::vector<float> sourceIntensity;

    // Geometry
    float sourceToIsocenter;  // source to isocenter distance [cm]
    float sourceToDetector;  // source to detector distance [cm]
    float xLen;  // length in x direction [cm]
    float yLen;  // length in y direction assuming the W thickness is 5cm [cm]
    float zLen;  // length in z direction default = 10cm

    // Bowtie and flat filter
    std::string bowtieType;  // Bowtie type: small, medium, large
    std::vector<std::string> flatFilterMat;
    std::vector<float> flatFilterThickness;

    // Directional quadrature set
    std::string quadType;
    int quadSpecial;
    int sn;  // N in Sn quadrature order
    int m;  // Total number of directions in all 8 octants

    // Cross section data set
    bool njoy;
    std::string njoyFile;
    int igm;  // Number of gamma energy groups
    int iht;  // Table position of the total cross section
    int ihs;  // Table position of the self-scatter cross section
    int ihm;  // Cross section table length per energy group (usually ihm = iht+igm)
    int ms;   // Cross section mixing table length (ms = 0 no mixing)
    int mtm;  // Total number of materials including al Pns (each Pn is considered a nuclide)
    int isct; // Maximum jorder of Legendre (ie scattering) expansion of the cross section (lowest is 0)

    // These two are used in my internal code, Dr. Liu uses xsection->msig
    std::vector<float> xsTot;
    std::vector<float> xsScat;

    //float *m_xsection;
    std::vector<float> xsection;

    // Source iteration data set
    float epsi;  // Convergence criteria
    float maxit;  // Max number of inner iterations (default = 20)
};

#endif // CONFIG_H
