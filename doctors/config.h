#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

class Config
{
public:
    Config();
    ~Config();

    void loadDefaults();

private:
    std::string m_resultsBasename;
    // callback output

    // Source and collimator
    // callback_spectrum
    std::string m_spectrumFile;
    float m_sourceFanAngle;
    float m_sourceConeAngle;
    float m_sourceTopGap;  // [cm]
    float m_sourceBottomGap; // [cm]
    float m_sourceLeftGap;
    float m_sourceRightGap;
    float m_sourceFrontGap;
    float m_sourceBackGap;
    float m_colXLen;  // [cm]  Collimator x length
    float m_colYLen;
    float m_colZLen;

    // Geometry
    float m_sourceToIsocenter;  // source to isocenter distance [cm]
    float m_sourceToDetector;  // source to detector distance [cm]
    float m_xLen;  // length in x direction [cm]
    float m_yLen;  // length in y direction assuming the W thickness is 5cm [cm]
    float m_zLen;  // length in z direction default = 10cm

    // Bowtie and flat filter
    std::string m_bowtieType;  // Bowtie type: small, medium, large
    //int m_flatFilterCount;  // The number of flat filters
    //float *m_flatFilterThickness;  // [cm]
    //std::string *m_flatFilterMat;  // material ('cu', 'al', etc.)
    std::vector<std::string> m_flatFilterMat;
    std::vector<float> m_flatFilterThickness;

    // Directional quadrature set
    int m_sn;  // N in Sn quadrature order
    int m_m;  // Total number of directions in all 8 octants

    // Cross section data set
    bool m_njoy;
    std::string m_njoyFile;
    int m_igm;  // Number of gamma energy groups
    int m_iht;  // Table position of the total cross section
    int m_ihs;  // Table position of the self-scatter cross section
    int m_ihm;  // Cross section table length per energy group (usually ihm = iht+igm)
    int m_ms;   // Cross section mixing table length (ms = 0 no mixing)
    int m_mtm;  // Total number of materials including al Pns (each Pn is considered a nuclide)
    int m_isct; // Maximum jorder of Legendre (ie scattering) expansion of the cross section (lowest is 0)
    //float *m_xsection;
    std::vector<float> m_xsection;

    // Source iteration data set
    float m_epsi;  // Convergence criteria
    float m_maxit;  // Max number of inner iterations (default = 20)



};

#endif // CONFIG_H
