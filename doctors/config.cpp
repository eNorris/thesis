#include "config.h"

#include <QDebug>

#include <cmath>

#include <libconfig.h++>

Config::Config() //: m_flatFilterThickness(NULL), m_flatFilterMat(NULL), m_xsection(NULL)
{
    // Initializes pointers to NULL and all other values to garbage
}

Config::~Config()
{

}

void Config::loadDefaults()
{
    resultsBasename = "test_input";

    // Source and collimator
    spectrumFile = "nsp_2.dat";
    sourceFanAngle = 50.0/180.0 * M_PI;
    sourceConeAngle = 2.0/180.0 * M_PI;
    sourceTopGap = 1.0;  // [cm]
    sourceBottomGap = 5.0; // [cm]
    sourceLeftGap = sourceBottomGap * tan(sourceFanAngle / 2.0);
    sourceRightGap = sourceLeftGap;
    sourceFrontGap = sourceBottomGap * tan(sourceConeAngle / 2.0);
    sourceBackGap = sourceFrontGap;
    colXLen = 10.0;  // [cm]  Collimator x length
    colYLen = 10.0;
    colZLen = 5.0;

    // Geometry
    sourceToIsocenter = 54.0;  // source to isocenter distance [cm]
    sourceToDetector = 95.0;  // source to detector distance [cm]
    xLen = 2.0 * sourceToDetector * tan(sourceFanAngle / 2.0);  // length in x direction [cm]
    yLen = sourceToDetector - sourceBottomGap + colYLen;  // length in y direction assuming the W thickness is 5cm [cm]
    zLen = 5.0;  // length in z direction default = 10cm

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
    xsection = {0.0, 0.0,  0.0002, 0.0002,     // Air x-section at 60 keV nearly void P0
                  0.0, 0.0,  0.0000, 0.0002,     // Air x-section P1 expansion
                  0.0, 0.0,  0.2059, 0.1770,     // Water x-section at 60 keV  P0
                  0.0, 0.0,  0.0000, 0.1770,     // Water x-section P1 expansion
                  0.0, 0.0, 71.4753, 0.0000,     // Tunsgten x-section at 60 keV P0
                  0.0, 0.0, 71.4753, 0.0000};    // Tunsgten x-section P1 expansion

    // Source iteration data set
    epsi = 5.0e-5;  // Convergence criteria
    maxit = 5;  // Max number of inner iterations (default = 20)
}

void Config::loadFile(std::string filename)
{
    libconfig::Config cfg;
    try{
        cfg.readFile(filename.c_str());
    } catch(const libconfig::FileIOException &fioex) {
        qDebug() << "I/O error reading file: " << QString::fromStdString(filename) << ": " << fioex.what();
        return;
    } catch(const libconfig::ParseException &pex) {
        qDebug() << "Parse error at " << pex.getFile() << ":" << pex.getLine() << " - " << pex.getError();
        return;
    }

    try{
        std::string name = cfg.lookup("name");
        qDebug() << "I read the phrase: " << QString::fromStdString(name) << "!";
    } catch(const libconfig::SettingNotFoundException &nfex){
        qDebug() << "No 'name' setting in config file";
        return;
    }

    const libconfig::Setting &root = cfg.getRoot();


}












