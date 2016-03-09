#include "config.h"

#include <cmath>

Config::Config() //: m_flatFilterThickness(NULL), m_flatFilterMat(NULL), m_xsection(NULL)
{
    // Initializes pointers to NULL and all other values to garbage
}

Config::~Config()
{
    // Release allocated heap memory
    /*
    if(m_flatFilterMat != NULL)
    {
        delete [] m_flatFilterThickness;
        m_flatFilterThickness = NULL;
    }

    if(m_flatFilterMat != NULL)
    {
        delete [] m_flatFilterMat;
        m_flatFilterMat = NULL;
    }

    if(m_xsection != NULL)
    {
        delete [] m_xsection;
        m_xsection = NULL;
    }
    */
}

void Config::loadDefaults()
{
    m_resultsBasename = "test_input";

    // Source and collimator
    m_spectrumFile = "nsp_2.dat";
    m_sourceFanAngle = 50.0/180.0 * M_PI;
    m_sourceConeAngle = 2.0/180.0 * M_PI;
    m_sourceTopGap = 1.0;  // [cm]
    m_sourceBottomGap = 5.0; // [cm]
    m_sourceLeftGap = m_sourceBottomGap * tan(m_sourceFanAngle / 2.0);
    m_sourceRightGap = m_sourceLeftGap;
    m_sourceFrontGap = m_sourceBottomGap * tan(m_sourceConeAngle / 2.0);
    m_sourceBackGap = m_sourceFrontGap;
    m_colXLen = 10.0;  // [cm]  Collimator x length
    m_colYLen = 10.0;
    m_colZLen = 5.0;

    // Geometry
    m_sourceToIsocenter = 54.0;  // source to isocenter distance [cm]
    m_sourceToDetector = 95.0;  // source to detector distance [cm]
    m_xLen = 2.0 * m_sourceToDetector * tan(m_sourceFanAngle / 2.0);  // length in x direction [cm]
    m_yLen = m_sourceToDetector - m_sourceBottomGap + m_colYLen;  // length in y direction assuming the W thickness is 5cm [cm]
    m_zLen = 5.0;  // length in z direction default = 10cm

    // Bowtie and flat filter
    m_bowtieType = "medium";  // Bowtie type: small, medium, large
    //m_flatFilterCount = 1;  // The number of flat filters
    //m_flatFilterThickness = new float[1];  // [cm]
    //m_flatFilterThickness[0] = 0.002;
    //m_flatFilterMat = new std::string[1];  // material ('cu', 'al', etc.)
    //m_flatFilterMat[0] = "cu";
    m_flatFilterThickness = {0.002f};  // Relies on C++11!
    m_flatFilterMat = {"cu"};

    // Directional quadrature set
    m_sn = 2;  // N in Sn quadrature order
    m_m = m_sn * (m_sn + 2);  // Total number of directions in all 8 octants

    // Cross section data set
    m_njoy = "no";
    m_njoyFile = "";
    m_igm = 1;  // Number of gamma energy groups
    m_iht = 3;  // Table position of the total cross section
    m_ihs = 4;  // Table position of the self-scatter cross section
    m_ihm = 4;  // Cross section table length per energy group (usually ihm = iht+igm)
    m_ms = 0;   // Cross section mixing table length (ms = 0 no mixing)
    m_mtm = 6;  // Total number of materials including al Pns (each Pn is considered a nuclide)
    m_isct = 1; // Maximum jorder of Legendre (ie scattering) expansion of the cross section (lowest is 0)
    m_xsection = {0.0, 0.0,  0.0002, 0.0002,     // Air x-section at 60 keV nearly void P0
                  0.0, 0.0,  0.0000, 0.0002,     // Air x-section P1 expansion
                  0.0, 0.0,  0.2059, 0.1770,     // Water x-section at 60 keV  P0
                  0.0, 0.0,  0.0000, 0.1770,     // Water x-section P1 expansion
                  0.0, 0.0, 71.4753, 0.0000,     // Tunsgten x-section at 60 keV P0
                  0.0, 0.0, 71.4753, 0.0000};    // Tunsgten x-section P1 expansion

    // Source iteration data set
    m_epsi = 5.0e-5;  // Convergence criteria
    m_maxit = 5;  // Max number of inner iterations (default = 20)
}














