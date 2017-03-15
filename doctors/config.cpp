#include "config.h"

#include <QDebug>
#include <qmath.h>

// This project relies on libconfig++ being installed. To install it, download libconfig-X.X.tar.gz from http://www.hyperrealm.com/libconfig/  unzip.
//   Follow the directions in the INSTALL file (./configure, make, make install). Then run "sudo ldconfig" to update the LD path variables so it can be found.
//#include <libconfig.h++>

Config::Config() //: m_flatFilterThickness(NULL), m_flatFilterMat(NULL), m_xsection(NULL)
{
    // Initializes pointers to NULL and all other values to garbage
}

Config::~Config()
{

}

void Config::loadDefaults()
{
	abort();

    /*
    resultsBasename = "test_input";

    // Source and collimator
    spectrumFile = "nsp_2.dat";
    sourceFanAngle = (float) 50.0/180.0 * M_PI;
    sourceConeAngle = (float) 2.0/180.0 * M_PI;
    sourceTopGap = 1.0f;  // [cm]
    sourceBottomGap = 5.0f; // [cm]
    sourceLeftGap = sourceBottomGap * tan(sourceFanAngle / 2.0f);
    sourceRightGap = sourceLeftGap;
    sourceFrontGap = sourceBottomGap * tan(sourceConeAngle / 2.0f);
    sourceBackGap = sourceFrontGap;
    colXLen = 10.0f;  // [cm]  Collimator x length
    colYLen = 10.0f;
    colZLen = 5.0f;

    sourceX.push_back(44.30);
    sourceY.push_back(4.55);
    sourceZ.push_back(2.5);
    sourceIntensity.push_back(1.0E6);

    // Geometry
    sourceToIsocenter = 54.0f;  // source to isocenter distance [cm]
    sourceToDetector = 95.0f;  // source to detector distance [cm]
    xLen = 2.0f * sourceToDetector * tan(sourceFanAngle / 2.0f);  // length in x direction [cm]
    yLen = sourceToDetector - sourceBottomGap + colYLen;  // length in y direction assuming the W thickness is 5cm [cm]
    zLen = 5.0f;  // length in z direction default = 10cm

    // Bowtie and flat filter
    bowtieType = "medium";  // Bowtie type: small, medium, large
    //m_flatFilterCount = 1;  // The number of flat filters
    //m_flatFilterThickness = new float[1];  // [cm]
    //m_flatFilterThickness[0] = 0.002;
    //m_flatFilterMat = new std::string[1];  // material ('cu', 'al', etc.)
    //m_flatFilterMat[0] = "cu";
    flatFilterThickness = {0.002f};  // Relies on C++11!
    flatFilterMat = {"cu"};

    // Directional quadrature set
    quadType = "sn";  // Type of quadrature
    quadSpecial = 2;  // Used for special quad types
    sn = 4;  // N in Sn quadrature order
    m = sn * (sn + 2);  // Total number of directions in all 8 octants

    // Cross section data set
    njoy = false;
    njoyFile = "";
    igm = 1;  // Number of gamma energy groups
    iht = 3;  // Table position of the total cross section
    ihs = 4;  // Table position of the self-scatter cross section
    ihm = 4;  // Cross section table length per energy group (usually ihm = iht+igm)
    ms = 0;   // Cross section mixing table length (ms = 0 no mixing)
    mtm = 6;  // Total number of materials including al Pns (each Pn is considered a nuclide)
    isct = 1; // Maximum order of Legendre (ie scattering) expansion of the cross section (lowest is 0)
    xsection = {0.0f, 0.0f,    0.0002f, 0.0002f,     // Air x-section at 60 keV nearly void P0
                  0.0f, 0.0f,  0.0000f, 0.0002f,     // Air x-section P1 expansion
                  0.0f, 0.0f,  0.2059f, 0.1770f,     // Water x-section at 60 keV  P0
                  0.0f, 0.0f,  0.0000f, 0.1770f,     // Water x-section P1 expansion
                  0.0f, 0.0f, 71.4753f, 0.0000f,     // Tunsgten x-section at 60 keV P0
                  0.0f, 0.0f, 71.4753f, 0.0000f};    // Tunsgten x-section P1 expansion

    // Total cross section
    xsTot = {0.0002f,      // Air
             0.2059f,      // Water
             71.4753f};    // Tungster
    // Scattering cross section
    xsScat = { 0.0002f,    // Air
               0.1770f,    // Water
               0.0001f};   // Tungster

    // Source iteration data set
    epsi = 5.0e-5f;  // Convergence criteria
    maxit = 20;  // Max number of inner iterations (default = 20)
    */
}












